//! Thresholds that belong to the machine rather than to taste.
//!
//! Every parallel path in the engine has a size below which entering a rayon
//! region costs more than it saves. Those sizes were measured — but on one laptop,
//! with twelve cores and 24 MB of L3. Written as bare numbers they are wrong
//! everywhere else, and wrong in the expensive direction: a server with 64 cores
//! would keep taking the serial path on work that should have been split.
//!
//! So they are expressed per thread, and multiplied by the threads rayon will
//! actually use. The model is "a thread needs this many rows before waking it is
//! worth the hand-off", which is what the measurements support: the aggregate's
//! crossover sat at ~178k rows on a machine reporting 20 threads, i.e. about 8.9k
//! rows each.
//!
//! **This is an extrapolation.** It is calibrated at one thread count and has
//! never been checked on a many-core machine. The competing model — that more
//! cores make parallelism pay off *sooner*, so the threshold should fall rather
//! than rise — is equally defensible from first principles, and only a run on such
//! a machine can settle it. Until then this at least moves with the hardware
//! instead of pretending every machine is this one.

use std::sync::OnceLock;

/// Threads rayon will use, read once.
///
/// `current_num_threads` is cheap but not free, and these thresholds are consulted
/// on every query; the value cannot change while the pool lives.
fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| rayon::current_num_threads().max(1))
}

/// Rows below which a parallel region is not worth entering.
///
/// `per_thread` is the calibrated share: how many rows one thread has to cover
/// before its wake-up pays for itself, for that particular kernel. A kernel doing
/// more work per row earns its hand-off sooner and so takes a smaller number.
#[inline]
pub(crate) fn par_min_rows(per_thread: usize) -> usize {
    per_thread.saturating_mul(threads())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_with_the_thread_count() {
        // Whatever this machine reports, the threshold has to be a multiple of it —
        // that is the whole property being relied on.
        let n = threads();
        assert_eq!(par_min_rows(1_000), 1_000 * n);
        assert!(par_min_rows(8_900) >= 8_900, "must never fall below one thread's share");
    }

    #[test]
    fn a_huge_share_does_not_wrap() {
        // Saturating, so a silly calibration turns into "never parallelise" rather
        // than into a tiny threshold that parallelises everything.
        assert_eq!(par_min_rows(usize::MAX), usize::MAX);
    }
}
