//! Translate every shader found in files or directories into the shader
//! cache a run reads, before the run: a first launch otherwise translates
//! all of them while the loading screen waits (`translate` reports what
//! they cannot handle; this one leaves the translations where the run
//! finds them).
//!
//! Entries are keyed by the translator, its environment settings, and the
//! host GPU facts (`victory-gpu::shader_cache`), so precompilation must run
//! with the same build and shader environment as the launch.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
#[cfg(windows)]
use victory_shader::emit::validator::Validator;
use victory_shader::{Container, FourCC};

use crate::progress::{Counter, Progress};
use crate::shader::{MAX_FILE, Source, sources};

/// Options controlling [`run`].
pub struct Options {
    pub paths: Vec<PathBuf>,
    pub host_wave: Option<u32>,
    pub jobs: Option<usize>,
    pub strict: bool,
    pub gui: bool,
    pub validate: bool,
    /// True when launched before a game run (`victory run --precompile`),
    /// allowing the user to close the progress window to skip and launch now.
    pub allow_skip: bool,
}

#[derive(Default)]
struct Tally {
    containers: usize,
    duplicates: usize,
    /// Translations the cache already held, and ones made now.
    cached: usize,
    translated: usize,
    /// With `--validate`: the emitted containers dxil.dll accepted, and
    /// those it did not, by its first line.
    validated: usize,
    invalid: BTreeMap<String, (usize, String)>,
    /// Containers that failed, by the translator's first line, with an
    /// example of one.
    failed: BTreeMap<String, (usize, String)>,
}

impl Tally {
    fn fail(&mut self, reason: &str, example: impl FnOnce() -> String) {
        let reason = reason.lines().next().unwrap_or_default().to_owned();
        self.failed
            .entry(reason)
            .or_insert_with(|| (0, example()))
            .0 += 1;
    }

    #[cfg(windows)]
    fn reject(&mut self, reason: &str, example: impl FnOnce() -> String) {
        let reason = reason.lines().next().unwrap_or_default().to_owned();
        self.invalid
            .entry(reason)
            .or_insert_with(|| (0, example()))
            .0 += 1;
    }

    fn merge(&mut self, other: Tally) {
        self.containers += other.containers;
        self.duplicates += other.duplicates;
        self.cached += other.cached;
        self.translated += other.translated;
        self.validated += other.validated;
        for (reason, (n, example)) in other.failed {
            self.failed
                .entry(reason)
                .or_insert_with(|| (0, example))
                .0 += n;
        }
        for (reason, (n, example)) in other.invalid {
            self.invalid
                .entry(reason)
                .or_insert_with(|| (0, example))
                .0 += n;
        }
    }
}

/// Walks `paths` and returns candidate `(path, file_size)` pairs, skipping
/// Victory's own cache/data directories and files larger than [`MAX_FILE`].
fn collect_candidates(
    paths: &[PathBuf],
    cache_dir: &Path,
    data_dir: &Path,
    cancelled: impl Fn() -> bool,
) -> Vec<(PathBuf, u64)> {
    let mut files = Vec::new();
    for root in paths {
        if cancelled() {
            break;
        }
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if cancelled() {
                break;
            }
            if !entry.file_type().is_file() {
                continue;
            }
            let len = entry.metadata().map_or(0, |m| m.len());
            if len == 0 || len > MAX_FILE {
                continue;
            }
            // Our own cache and saves can sit inside the game folder, and
            // their files are not the title's shaders.
            let path = entry.path();
            if path.starts_with(cache_dir) || path.starts_with(data_dir) {
                continue;
            }
            files.push((path.to_path_buf(), len));
        }
    }
    files
}

pub fn run(options: Options) -> Result<()> {
    let paths = crate::default_paths(options.paths)?;
    // Query the host GPU first so the host D3D12 device (and `host_rdna` /
    // `view_heap_limit`) is initialized before any `--host-wave` override is
    // applied; otherwise the first `precompile_shader` call would overwrite
    // the override with the real card's wave width.
    let detected_wave = victory_hle::modules::d3d12x::host_wave_max();
    let wave = match options.host_wave {
        Some(wave) => wave,
        None => detected_wave.context(
            "no host Direct3D 12 device to ask its widest wave; pass --host-wave \
             (32 for NVIDIA, 64 for AMD RDNA)",
        )?,
    };
    anyhow::ensure!(
        wave.is_power_of_two() && (4..=128).contains(&wave),
        "{wave} is not a wave size (4 to 128, a power of two)"
    );
    victory_shader::emit::set_host_max_wave(wave);
    if options.host_wave.is_some() && wave < 64 {
        victory_shader::emit::set_host_rdna(false);
    }

    let cfg = victory_core::Config::load()?;
    let dir = cfg.cache_subdir("shaders")?;
    victory_hle::modules::d3d12x::use_shader_cache(dir.clone());
    println!("host wave    {wave} lanes");
    println!("shader cache {}", dir.display());

    #[cfg(windows)]
    let dll = options.validate.then(Validator::default_path).flatten();
    #[cfg(windows)]
    if options.validate {
        Validator::load(dll.as_deref()).map_err(anyhow::Error::msg)?;
        println!(
            "validator    {}",
            dll.as_ref()
                .map_or("dxil.dll".into(), |p| p.display().to_string())
        );
    }
    #[cfg(not(windows))]
    if options.validate {
        anyhow::bail!("--validate requires Windows (dxil.dll)");
    }

    let progress = options
        .gui
        .then(|| Progress::new("VIctory — Precompiling Shaders", options.allow_skip));

    let started = Instant::now();
    let threads = options
        .jobs
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
        .max(1);

    let files = collect_candidates(&paths, &cfg.cache_dir, &cfg.data_dir, || {
        progress.as_ref().is_some_and(Progress::is_cancelled)
    });
    let total_files = files.len() as u64;
    let total_bytes: u64 = files.iter().map(|(_, len)| *len).sum();
    if let Some(p) = &progress {
        p.set_totals(total_files, total_bytes);
    }

    let seen = Mutex::new(HashSet::new());
    let done = AtomicUsize::new(0);
    let total = Mutex::new(Tally::default());

    for (idx, (path, file_len)) in files.iter().enumerate() {
        if progress.as_ref().is_some_and(Progress::is_cancelled) {
            break;
        }
        if let Some(p) = &progress {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default();
            p.set_status(format!(
                "Precompiling shaders — {name} ({}/{total_files})",
                idx + 1
            ));
        }
        let Ok(data) = std::fs::read(path) else {
            if let Some(p) = &progress {
                p.add_bytes(*file_len);
                p.finish_file();
            }
            continue;
        };
        let containers = sources(&data);
        if containers.is_empty() {
            if let Some(p) = &progress {
                p.add_bytes(*file_len);
                p.finish_file();
            }
            continue;
        }

        let per_container = *file_len / containers.len() as u64;
        let remainder = *file_len % containers.len() as u64;
        let next = AtomicUsize::new(0);

        std::thread::scope(|s| {
            for _ in 0..threads.min(containers.len()) {
                s.spawn(|| {
                    let mut t = Tally::default();
                    let counter = progress.as_ref().map(Progress::counter);
                    #[cfg(windows)]
                    let mut validator = options
                        .validate
                        .then(|| Validator::load(dll.as_deref()).ok())
                        .flatten();
                    #[cfg(not(windows))]
                    let mut validator: Option<()> = None;

                    loop {
                        if counter.as_ref().is_some_and(Counter::is_cancelled) {
                            break;
                        }
                        let Some(source) = containers.get(next.fetch_add(1, Ordering::Relaxed))
                        else {
                            break;
                        };
                        let outcome = one(source, path, &seen, &mut t, &mut validator);
                        if let Some(counter) = &counter {
                            if let Some(hit) = outcome.outcome() {
                                counter.container(hit);
                            }
                            counter.add_bytes(per_container);
                        }
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_multiple_of(1000) {
                            println!(
                                "  {n} containers, {:.0} a second",
                                n as f64 / started.elapsed().as_secs_f64().max(1e-3)
                            );
                        }
                    }
                    lock(&total).merge(t);
                });
            }
        });

        if let Some(p) = &progress {
            if !p.is_cancelled() {
                p.add_bytes(remainder);
            }
            p.finish_file();
        }
    }

    let cancelled = progress.as_ref().is_some_and(Progress::is_cancelled);
    drop(progress);

    let t = total.into_inner().unwrap_or_else(|e| e.into_inner());
    if cancelled {
        println!(
            "precompile   skipped by user after {:.1} s (already-cached shaders were kept)",
            started.elapsed().as_secs_f64()
        );
    }
    println!(
        "containers   {} ({} duplicates skipped)",
        t.containers, t.duplicates
    );
    println!(
        "translated   {} stored now, {} already in the cache, {} failed",
        t.translated,
        t.cached,
        t.failed.values().map(|f| f.0).sum::<usize>()
    );
    let mut failed: Vec<_> = t.failed.into_iter().collect();
    failed.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (reason, (n, example)) in &failed {
        println!("  {n:>7}  {reason}  (e.g. {example})");
    }
    if options.validate {
        println!(
            "DXIL         {} of {} validated",
            t.validated,
            t.validated + t.invalid.values().map(|f| f.0).sum::<usize>()
        );
        let mut invalid: Vec<_> = t.invalid.into_iter().collect();
        invalid.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        for (reason, (n, example)) in &invalid {
            println!("  {n:>7}  {reason}  (e.g. {example})");
        }
    }
    println!(
        "cache        {} entries, {:.0} MB; {:.1} s",
        count(&dir),
        bytes(&dir) as f64 / (1 << 20) as f64,
        started.elapsed().as_secs_f64()
    );
    if options.strict && !cancelled && !failed.is_empty() {
        anyhow::bail!("{} containers failed", failed.len());
    }
    Ok(())
}

/// What became of one container, for the progress window.
enum Done {
    /// Stored now (`false`), or already in the cache (`true`).
    Counted(bool),
    /// It could not be translated.
    Failed,
    /// Not a container to translate (a duplicate or a corrupt record).
    Skipped,
}

impl Done {
    fn outcome(&self) -> Option<Option<bool>> {
        match self {
            Done::Counted(hit) => Some(Some(*hit)),
            Done::Failed => Some(None),
            Done::Skipped => None,
        }
    }
}

#[cfg(windows)]
type OptValidator = Option<Validator>;
#[cfg(not(windows))]
type OptValidator = Option<()>;

/// Translates one container into the cache, counting what it found there.
fn one(
    source: &Source,
    path: &Path,
    seen: &Mutex<HashSet<blake3::Hash>>,
    t: &mut Tally,
    validator: &mut OptValidator,
) -> Done {
    let example = || format!("{}{}", path.display(), source.label());
    let Some(bytes) = source.bytes() else {
        t.fail("corrupt record", example);
        return Done::Skipped;
    };
    t.containers += 1;
    let Ok(container) = Container::parse(&bytes) else {
        t.fail("not a DXBC container", example);
        return Done::Skipped;
    };
    // The run translates a container as the title passes it; the same DXIL
    // (or ACCC) in another container is the same translation.
    let Some(dxil) = container
        .part(FourCC::DXIL)
        .or(container.part(FourCC::ACCC))
    else {
        t.fail("no DXIL part (SM5 or older)", example);
        return Done::Skipped;
    };
    if !lock(seen).insert(blake3::hash(dxil.data)) {
        t.duplicates += 1;
        return Done::Skipped;
    }
    match victory_hle::modules::d3d12x::precompile_shader(&bytes) {
        Ok((dxil, hit)) => {
            if hit {
                t.cached += 1;
            } else {
                t.translated += 1;
            }
            #[cfg(windows)]
            if let Some(validator) = validator {
                let mut dxil = dxil;
                match validator.validate(&mut dxil) {
                    Ok(()) => t.validated += 1,
                    Err(e) => t.reject(&e, example),
                }
            }
            #[cfg(not(windows))]
            let _ = (&dxil, validator);
            Done::Counted(hit)
        }
        Err(e) => {
            t.fail(&e, example);
            Done::Failed
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn count(dir: &Path) -> usize {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .count()
}

fn bytes(dir: &Path) -> u64 {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_merge_combines_counts_and_failure_reasons() {
        let mut a = Tally {
            containers: 3,
            duplicates: 1,
            cached: 1,
            translated: 1,
            validated: 2,
            ..Default::default()
        };
        a.fail("unsupported op\ndetails", || "file_a@0".into());

        let mut b = Tally {
            containers: 4,
            duplicates: 2,
            cached: 2,
            translated: 0,
            validated: 1,
            ..Default::default()
        };
        b.fail("unsupported op\nother", || "file_b@1".into());
        b.fail("bad layout", || "file_b@2".into());

        a.merge(b);
        assert_eq!(a.containers, 7);
        assert_eq!(a.duplicates, 3);
        assert_eq!(a.cached, 3);
        assert_eq!(a.translated, 1);
        assert_eq!(a.validated, 3);
        assert_eq!(a.failed.get("unsupported op").map(|v| v.0), Some(2));
        assert_eq!(a.failed.get("bad layout").map(|v| v.0), Some(1));
    }

    #[test]
    fn collect_candidates_skips_cache_data_and_empty_files() {
        let root = std::env::temp_dir().join(format!(
            "victory-precompile-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let cache_dir = root.join("cache");
        let data_dir = root.join("data");
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::fs::create_dir_all(&data_dir).unwrap();

        let game_file = root.join("shaders.bin");
        std::fs::write(&game_file, b"DXBC_dummy").unwrap();
        std::fs::write(root.join("empty.bin"), b"").unwrap();
        std::fs::write(cache_dir.join("cached.bin"), b"ignored").unwrap();
        std::fs::write(data_dir.join("save.bin"), b"ignored").unwrap();

        let found = collect_candidates(
            std::slice::from_ref(&root),
            &cache_dir,
            &data_dir,
            || false,
        );
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, game_file);
        assert_eq!(found[0].1, 10);
    }

    #[test]
    fn done_outcome_maps_for_progress_counter() {
        assert_eq!(Done::Counted(true).outcome(), Some(Some(true)));
        assert_eq!(Done::Counted(false).outcome(), Some(Some(false)));
        assert_eq!(Done::Failed.outcome(), Some(None));
        assert_eq!(Done::Skipped.outcome(), None);
    }
}
