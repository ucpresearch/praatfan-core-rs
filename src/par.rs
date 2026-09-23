//! Order-preserving parallel map over frames, with a serial fallback.
//!
//! Every analysis computes its global quantities (peak, window, FFT plans)
//! serially, then maps an independent per-frame computation over the frame
//! indices. [`map_init`] runs that map on rayon when the `parallel` feature is
//! on (and the target is not wasm32), and as a plain loop otherwise. Each
//! frame's arithmetic is unchanged and results are collected in index order,
//! so output is bit-identical regardless of thread count.
//!
//! `init` builds per-worker scratch. rayon calls it once per split job, not
//! once per thread or per item, so callers must fully overwrite scratch on
//! every item (exactly what the serial loop, which reuses one scratch for all
//! items, already requires).
//!
//! # Thread pool and fork()
//!
//! rayon's global pool does not survive `fork()`: in the child the pool's
//! threads are gone and the first parallel call deadlocks. Python's
//! `multiprocessing` forks by default on Linux, so we never use the global
//! pool. Instead we own a pool tagged with the process id that built it, and
//! rebuild it when the pid changes (leaking the stale one, since dropping it
//! would try to join threads that do not exist in the child). If the caller
//! is already running on some rayon worker, we use that pool, so callers who
//! install their own pool keep control. The pool size honours
//! `RAYON_NUM_THREADS`.

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
mod imp {
    use rayon::prelude::*;
    use rayon::{ThreadPool, ThreadPoolBuilder};
    use std::sync::{Arc, Mutex};

    static POOL: Mutex<Option<(u32, Arc<ThreadPool>)>> = Mutex::new(None);

    fn pool() -> Arc<ThreadPool> {
        let pid = std::process::id();
        let mut guard = POOL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((owner, pool)) = guard.as_ref() {
            if *owner == pid {
                return Arc::clone(pool);
            }
        }
        if let Some((_, stale)) = guard.take() {
            // Built by the parent before fork(); its threads don't exist here.
            std::mem::forget(stale);
        }
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .thread_name(|i| format!("praatfan-{i}"))
                .build()
                .expect("failed to build praatfan thread pool"),
        );
        *guard = Some((pid, Arc::clone(&pool)));
        pool
    }

    /// Run `f` inside the crate's pool (or the caller's, if already on one).
    pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
        if rayon::current_thread_index().is_some() {
            f()
        } else {
            pool().install(f)
        }
    }

    pub fn map_init<T, S, I, F>(n: usize, init: I, f: F) -> Vec<T>
    where
        T: Send,
        I: Fn() -> S + Sync + Send,
        F: Fn(&mut S, usize) -> T + Sync + Send,
    {
        install(|| (0..n).into_par_iter().map_init(&init, |s, i| f(s, i)).collect())
    }
}

#[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
mod imp {
    pub fn map_init<T, S, I, F>(n: usize, init: I, f: F) -> Vec<T>
    where
        I: Fn() -> S,
        F: Fn(&mut S, usize) -> T,
    {
        let mut scratch = init();
        (0..n).map(|i| f(&mut scratch, i)).collect()
    }
}

/// `(0..n).map(|i| f(&mut scratch, i)).collect()`, in parallel when enabled.
pub(crate) use imp::map_init;

/// Map `f` over `items` in order, in parallel when enabled.
pub(crate) fn map<A, T, F>(items: &[A], f: F) -> Vec<T>
where
    A: Sync,
    T: Send,
    F: Fn(&A) -> T + Sync + Send,
{
    map_init(items.len(), || (), |_, i| f(&items[i]))
}
