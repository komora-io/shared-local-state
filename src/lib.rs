use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use parking_lot::{Condvar, Mutex, RwLock};

type Id = u64;

/// Facilitates sharing of some local state with other threads.
///
/// # Examples
///
/// Maintain a counter of concurrent threads:
/// ```
/// use shared_local_state::SharedLocalState;
///
/// let sls = SharedLocalState::new(());
/// assert_eq!(sls.len(), 1);
///
/// // adds a new shared state, causing the count to grow to 2
/// let sls_2 = sls.insert(());
///
/// // signal 1
/// let (tx1, rx1) = std::sync::mpsc::channel::<()>();
/// // signal 2
/// let (tx2, rx2) = std::sync::mpsc::channel::<()>();
///
/// std::thread::spawn(move || {
///     // perform some work with the shared state in another thread
///     sls_2.update_and_notify(|state| assert_eq!(*state, ()));
///
///     // wait for signal 1 which lets us clean up
///     for _ in rx1 {}
///
///     // remove shared state, causing the number of shared
///     // states to drop back to 1.
///     drop(sls_2);
///
///     // send signal 2, telling the main thread that we have
///     // cleaned up our shared local state.
///     drop(tx2);
/// });
///
/// assert_eq!(sls.len(), 2);
///
/// // send signal 1, telling the spawned thread they can clean up
/// drop(tx1);
///
/// // wait for signal 2, when we know the spawned thread has cleaned up
/// for _ in rx2 {}
///
/// assert_eq!(sls.len(), 1);
/// ```
#[derive(Debug)]
pub struct SharedLocalState<T> {
    shared_state: Arc<SharedState<T>>,
    id: Id,
    state: Arc<T>,
}

impl<T> Drop for SharedLocalState<T> {
    fn drop(&mut self) {
        let mut registry = self.shared_state.registry.write();
        registry
            .remove(&self.id)
            .expect("must be able to remove registry's shared state on drop");
    }
}

const INITIAL_ID: u64 = 0;

impl<T> SharedLocalState<T> {
    /// Create a new shared registry that makes the provided local state
    /// visible to other [`SharedLocalState`] handles created through the
    /// [`insert`] method.
    ///
    /// If the returned [`SharedLocalState`] object is dropped, the shared state
    /// will be removed from the shared registry and dropped as well.
    ///
    /// [`insert`]: SharedLocalState::insert
    pub fn new(state: T) -> SharedLocalState<T> {
        let arc = Arc::new(state);
        let registry = RwLock::new([(INITIAL_ID, arc.clone())].into());

        let shared_state = Arc::new(SharedState {
            registry,
            mu: Mutex::new(()),
            cv: Condvar::new(),
        });

        SharedLocalState {
            id: INITIAL_ID,
            shared_state,
            state: arc,
        }
    }

    /// Registers some local state for the rest of the [`SharedLocalState`]
    /// handles to access.
    ///
    /// If the returned [`SharedLocalState`] object is dropped, the shared state
    /// will be removed from the shared registry and dropped.
    pub fn insert(&self, state: T) -> SharedLocalState<T> {
        static IDGEN: AtomicU64 = AtomicU64::new(INITIAL_ID + 1);

        // Ordering not important, only uniqueness
        // which is still guaranteed w/ Relaxed.
        let id = IDGEN.fetch_add(1, Ordering::Relaxed);

        let arc = Arc::new(state);

        self.shared_state.registry.write().insert(id, arc.clone());

        // Broadcast to waiters that a new handle exists.
        self.notify_all();

        SharedLocalState {
            id,
            shared_state: self.shared_state.clone(),
            state: arc,
        }
    }

    /// The number of shared states associated with this
    /// [`SharedLocalState`]. This will always be non-zero
    /// because the existence of a single [`SharedLocalState`]
    /// implies the existence of at least one shared state.
    pub fn len(&self) -> usize {
        self.shared_state.registry.read().len()
    }

    /// Update the local shared state and notify any other threads
    /// that may be waiting on updates via the [`find_or_wait`] method.
    /// Only makes sense if `T` is `Sync` because it must be accessed
    /// through an immutable reference. If you want to minimize
    /// the underlying `Condvar` notification effort, or if
    /// you are only interested in viewing the shared
    /// local state, use [`access_without_notification`] instead.
    ///
    /// [`find_or_wait`]: SharedLocalState::find_or_wait
    /// [`access_without_notification`]: SharedLocalState::access_without_notification
    pub fn update_and_notify<F, R>(&self, f: F) -> R
    where
        F: Fn(&T) -> R,
    {
        let ret = f(&self.state);
        self.notify_all();
        ret
    }

    /// Accesses the shared local state without notifying other
    /// threads that may be waiting for updates in concurrent calls
    /// to [`find_or_wait`].
    ///
    /// [`find_or_wait`]: SharedLocalState::find_or_wait
    pub fn access_without_notification<F, R>(&self, f: F) -> R
    where
        F: Fn(&T) -> R,
    {
        f(&self.state)
    }

    /// Ensures that any modifications performed via [`access_without_notification`]
    /// are visible to threads waiting for updates in concurrent calls
    /// to [`find_or_wait`].
    ///
    /// [`access_without_notification`]: SharedLocalState::access_without_notification
    /// [`find_or_wait`]: SharedLocalState::find_or_wait
    pub fn notify_all(&self) {
        // it is important to acquire the cv's associated
        // mutex to linearize notifications with anyone
        // who may be waiting on an update in `get_or_wait`
        drop(self.shared_state.mu.lock());

        self.shared_state.cv.notify_all();
    }

    /// Iterates over all shared states until the provided `F` returns
    /// `Some(R)`, which is then returned from this method. If `F` does
    /// not return `Some(R)` for any shared state, a condition variable
    /// is used to avoid spinning until shared states have been modified.
    pub fn find_or_wait<F, R>(&self, f: F) -> R
    where
        F: Fn(&T) -> Option<R>,
    {
        // first try
        {
            let registry = self.shared_state.registry.read();

            for state in registry.values() {
                if let Some(r) = f(state) {
                    return r;
                }
            }
        }

        // now take out lock and do it again in a loop,
        // blocking on the condvar if nothing is found
        let mut mu = self.shared_state.mu.lock();

        loop {
            let registry = self.shared_state.registry.read();

            for state in registry.values() {
                if let Some(r) = f(state) {
                    return r;
                }
            }

            drop(registry);

            self.shared_state.cv.wait(&mut mu);
        }
    }

    /// Folds over all shared local states.
    pub fn fold<B, F>(&self, init: B, f: F) -> B
    where
        F: FnMut(B, &T) -> B,
    {
        let registry = self.shared_state.registry.read();
        registry.values().map(|v| &**v).fold(init, f)
    }

    /// Maps over all shared local states.
    pub fn map<B, F, R>(&self, mut f: F) -> R
    where
        F: FnMut(&T) -> B,
        R: FromIterator<B>,
    {
        let registry = self.shared_state.registry.read();
        registry.values().map(|v| f(v)).collect()
    }

    /// Filter-maps over all shared local states.
    pub fn filter_map<B, F, R>(&self, mut f: F) -> R
    where
        F: FnMut(&T) -> Option<B>,
        R: FromIterator<B>,
    {
        let registry = self.shared_state.registry.read();
        registry.values().filter_map(|v| f(v)).collect()
    }
}

#[derive(Debug)]
struct SharedState<T> {
    registry: RwLock<HashMap<Id, Arc<T>>>,
    mu: Mutex<()>,
    cv: Condvar,
}
