use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Condvar, Mutex, RwLock,
};

type Id = u64;

#[derive(Debug, Default)]
pub struct Registry<T> {
    shared_state: Arc<SharedState<T>>,
    id: Id,
}

impl<T> Drop for Registry<T> {
    fn drop(&mut self) {
        let mut registry = self.shared_state.registry.write().unwrap();
        registry
            .remove(&self.id)
            .expect("must be able to remove registry's shared state on drop");
    }
}

impl<T> Registry<T> {
    /// Create a new registry handle that has the provided local state.
    pub fn clone(&self, state: T) -> Registry<T> {
        static IDGEN: AtomicU64 = AtomicU64::new(1);

        // Ordering not important, only uniqueness
        // which is still guaranteed w/ Relaxed.
        let id = IDGEN.fetch_add(1, Ordering::Relaxed);

        self.shared_state
            .registry
            .write()
            .unwrap()
            .insert(id, state);

        // Broadcast to waiters that a new handle exists.
        self.shared_state.cv.notify_all();

        Registry {
            id,
            shared_state: self.shared_state.clone(),
        }
    }

    pub fn get_or_wait<F, R>(&self, f: F) -> R
    where
        F: Fn(&T) -> Option<R>,
    {
        // first try
        {
            let registry = self.shared_state.registry.read().unwrap();

            for state in registry.values() {
                if let Some(r) = f(state) {
                    return r;
                }
            }
        }

        // now take out lock and do it again in a loop,
        // blocking on the condvar if nothing is found
        let mut mu = self.shared_state.mu.lock().unwrap();

        loop {
            let registry = self.shared_state.registry.read().unwrap();

            for state in registry.values() {
                if let Some(r) = f(state) {
                    return r;
                }
            }

            drop(registry);

            mu = self.shared_state.cv.wait(mu).unwrap();
        }
    }

    pub fn fold<B, F>(&self, init: B, f: F) -> B
    where
        F: FnMut(B, &T) -> B,
    {
        let registry = self.shared_state.registry.read().unwrap();
        registry.values().fold(init, f)
    }

    pub fn map<B, F, R>(&self, f: F) -> R
    where
        F: FnMut(&T) -> B,
        R: FromIterator<B>,
    {
        let registry = self.shared_state.registry.read().unwrap();
        registry.values().map(f).collect()
    }

    pub fn notify_all(&self) {
        self.shared_state.cv.notify_all();
    }
}

#[derive(Debug, Default)]
struct SharedState<T> {
    registry: RwLock<HashMap<Id, T>>,
    mu: Mutex<()>,
    cv: Condvar,
}
