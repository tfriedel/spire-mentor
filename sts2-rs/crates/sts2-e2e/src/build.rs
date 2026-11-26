//! The e2e lane build (`ppo.e2e.build`, GH #1535): which build's tables the
//! tokenizer reads.
//!
//! A training process plays one lane build. `STS2_E2E_GAME_VERSION` names it
//! before the process starts, as on the Python side, and is read once; unset
//! (every v0.111.0 lane) it is v0.111.0, so those read exactly the tables they
//! always did.
//!
//! A host that serves several nets (the workbench, the mod's Instinct advisor:
//! the all-character v0.111.0 net and the Fool's v0.113.0 one) picks the tables
//! per net instead: [`Scope::enter`] sets this thread's build from the net's
//! header (`E2eNet::build`) until the guard drops, and `tokens::encode_for`
//! enters it on every row. Each build's tables are built once, in their own
//! `OnceLock`, so one process holds both. A later build reads its own content-ID
//! table (`data/vocab_v0113.tsv`: `vocab_v0111.tsv`'s rows, then the new names),
//! card types and hit counts, and every build's registries for the S_ctx content
//! tables (`card_any`, ...), so its new content has rows of its own.

use std::cell::Cell;
use std::sync::OnceLock;

use sts2_core::GameVersion;

/// The environment variable that names the lane build (`build.BUILD_ENV`).
pub const BUILD_ENV: &str = "STS2_E2E_GAME_VERSION";

/// `build.resolve`: the lane build `raw` names (`None` or empty: v0.111.0).
///
/// # Errors
///
/// A build no e2e lane plays (an unknown one, or v0.107.1).
pub fn resolve(raw: Option<&str>) -> Result<GameVersion, String> {
    let raw = raw.map(str::trim).unwrap_or_default();
    if raw.is_empty() {
        return Ok(GameVersion::V0_111_0);
    }
    let with_v = if raw.starts_with('v') {
        raw.to_owned()
    } else {
        format!("v{raw}")
    };
    let gv: GameVersion = with_v
        .parse()
        .map_err(|e| format!("{BUILD_ENV}={raw:?}: {e}"))?;
    match gv {
        GameVersion::V0_111_0 | GameVersion::V0_113_0 => Ok(gv),
        other @ GameVersion::V0_107_1 => Err(format!(
            "{BUILD_ENV}={raw:?}: an e2e lane does not play {other}"
        )),
    }
}

/// This process's lane build (read once from [`BUILD_ENV`]).
///
/// # Panics
///
/// If [`BUILD_ENV`] names a build no lane plays.
#[must_use]
pub fn lane() -> GameVersion {
    static B: OnceLock<GameVersion> = OnceLock::new();
    *B.get_or_init(|| {
        resolve(std::env::var(BUILD_ENV).ok().as_deref()).unwrap_or_else(|e| panic!("{e}"))
    })
}

thread_local! {
    /// The build a [`Scope`] set on this thread, if any.
    static SCOPED: Cell<Option<GameVersion>> = const { Cell::new(None) };
}

/// The build whose tables this thread reads: the innermost [`Scope`]'s, else
/// the process's [`lane`].
#[must_use]
pub fn current() -> GameVersion {
    SCOPED.with(Cell::get).unwrap_or_else(lane)
}

/// Whether the current build ([`current`]) is after v0.111.0.
#[must_use]
pub fn later() -> bool {
    current() != GameVersion::V0_111_0
}

/// The lane build's spelling (`factory.GAME_VERSION`).
#[must_use]
pub fn game_version() -> &'static str {
    lane().as_str()
}

/// Sets this thread's table build until dropped, then restores the previous
/// one (scopes nest).
#[derive(Debug)]
#[must_use = "the build reverts when the guard drops"]
pub struct Scope {
    prev: Option<GameVersion>,
}

impl Scope {
    /// Read `gv`'s tables on this thread while the guard lives.
    ///
    /// # Panics
    ///
    /// If no e2e lane plays `gv` (v0.107.1, an unknown build).
    pub fn enter(gv: GameVersion) -> Self {
        let gv = resolve(Some(gv.as_str())).unwrap_or_else(|e| panic!("{e}"));
        Self {
            prev: SCOPED.with(|s| s.replace(Some(gv))),
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        SCOPED.with(|s| s.set(self.prev));
    }
}

/// One value per lane build, each built on first use (the build's tables).
pub(crate) struct PerBuild<T> {
    v0111: OnceLock<T>,
    v0113: OnceLock<T>,
}

impl<T> PerBuild<T> {
    pub(crate) const fn new() -> Self {
        Self {
            v0111: OnceLock::new(),
            v0113: OnceLock::new(),
        }
    }

    /// The current build's value, built by `f` (under that build) on first use.
    pub(crate) fn get(&self, f: impl FnOnce() -> T) -> &T {
        if later() { &self.v0113 } else { &self.v0111 }.get_or_init(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_names_a_lane_build() {
        assert_eq!(resolve(None), Ok(GameVersion::V0_111_0));
        assert_eq!(resolve(Some("")), Ok(GameVersion::V0_111_0));
        assert_eq!(resolve(Some("v0.113.0")), Ok(GameVersion::V0_113_0));
        assert_eq!(resolve(Some("0.113.1")), Ok(GameVersion::V0_113_0));
        assert!(resolve(Some("v0.107.1")).is_err());
        assert_eq!(resolve(Some("v0.114.0")), Ok(GameVersion::V0_113_0));
        assert!(resolve(Some("v0.115.0")).is_err());
    }

    #[test]
    fn scopes_nest_and_restore() {
        let base = current();
        {
            let _a = Scope::enter(GameVersion::V0_113_0);
            assert!(later());
            {
                let _b = Scope::enter(GameVersion::V0_111_0);
                assert!(!later());
            }
            assert_eq!(current(), GameVersion::V0_113_0);
        }
        assert_eq!(current(), base);
    }
}
