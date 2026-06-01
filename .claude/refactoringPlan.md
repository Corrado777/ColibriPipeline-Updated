# ColibriPipeline Refactoring Plan

**Author / context:** Plan developed in collaboration with Claude based on inspection of the `ColibriPipeline` and `ColibriObservatory/PipelineAutomation` repositories.

**Goal:** Make the per-telescope reduction pipeline run robustly to completion every night, while setting the stage for a longer-term production-quality refactor. Work is staged so that nightly observing is not disrupted by the larger cleanup.

**Audience:** Future maintainers of ColibriPipeline / ColibriObservatory. Treat this document as the agreed roadmap; deviations should update it.

---

## 0. Diagnosis of the current "stuck pipeline" bug

The orchestration lives in `ColibriObservatory/PipelineAutomation/pipeline_automation.py`. It runs each stage (`colibri_main_py3`, `coordsfinder`, `image_stats_dark`, `sensitivity`, `wcsmatching`, `simultaneous_occults`, `colibri_secondary`, `cumulative_stats`, `timeline`, `email_timeline`) by spawning a `subprocess.run`. Completion is tracked by writing a per-stage sentinel `*.txt` file ("stop file") into either the night's raw-data directory or its archive directory. Cross-telescope handoffs (Green waiting for Red+Blue, Blue waiting for Red+Green) are tracked by additional sentinels (`done.txt`, `timeline_ready.txt`, `generate_artificial.txt`).

This scheme has six interacting failure modes, in roughly decreasing order of severity:

1. **Sentinel ≠ success.** In `runProcesses`, the sentinel is written inside the `try` block immediately after `subprocess.run()` returns. The exit code is never inspected. A science script that crashes mid-run still leaves a sentinel saying "done," so the next run skips it forever.
2. **Unbounded blocking cross-telescope waits.** Patterns like `while not (path.is_file() == path.parent.is_dir()): time.sleep(300)` have no timeout and no fallback. A single telescope failing mid-night freezes the other two indefinitely.
3. **`is_file() == is_dir()` is the wrong predicate.** When neither the file nor the parent directory exist, `False == False == True` and the wait exits early — i.e. a telescope races ahead before its peers have even created the night directory.
4. **Two parallel "have I done this?" systems.** `colibri_main_py3.py` already independently decides which minute directories to skip based on the presence of `*_pos.npy` files in the archive. The orchestrator's sentinel scheme is a second, disjoint mechanism. They can — and do — disagree.
5. **Sentinel locations are scattered.** Some live in `D:/ColibriData/<YYYYMMDD>/`, others in `D:/ColibriArchive/<YYYY-MM-DD>/`, others in subdirectories. `clean_stopfiles.py` only sweeps the data directory.
6. **`processdata.py` is dead but present.** It implements a slightly different sentinel scheme and is referenced in `ProcessColibriData.bat` (commented out). It should be deleted to avoid confusion.

The "stuck" symptom most often comes from (1) + (2): a science script silently fails, the sentinel is written, the orchestrator advances, eventually some downstream stage waits for a peer that will never arrive (because the peer is also stuck on the same problem one stage further back, or has crashed entirely).

---

## 1. Staged plan

### Phase 0 — Simulated test environment (do first; ~1–2 days)

This must come before any code changes so that fixes can be exercised without touching live telescope data.

**Goals**
- Reproduce the on-telescope filesystem layout on a development machine.
- Allow the orchestrator and the science scripts to run end-to-end on a tiny synthetic night.
- Allow the cross-telescope synchronization paths (`R:/`, `G:/`, `B:/`) to be mocked.

**Layout to mirror**

```
<base>/
├── ColibriData/
│   └── <YYYYMMDD>/
│       ├── Dark/<YYYYMMDD_HH.MM.SS.fff>/  ← bias frames
│       └── <YYYYMMDD_HH.MM.SS.fff>/        ← minute dirs of .rcd
├── ColibriArchive/
│   └── <YYYY-MM-DD>/                       ← per-night outputs
├── ColibriImages/
├── CentralRepo/CumulativeStats/
├── Logs/
│   ├── ACP/<YYYYMMDD>-ACP.log
│   ├── Pipeline/
│   └── Weather/Weather/
└── tmp/
```

**Approach (recommended)**

- Introduce a single configuration knob `COLIBRI_BASE` (env var, optionally overridable on the command line). Every script currently hardcoding `pathlib.Path('D:/')` reads from this instead.
- For Phase 0 we do **not** need to refactor every script — instead, monkeypatch `BASE_PATH` from a shim at the top of each script the simulator drives, OR run on a machine where `D:/` is `subst`'d to a working directory (Windows) or symlinked (Linux/Mac).
- For peer-telescope mounts: spin up three sibling directories (`<base>_red`, `<base>_green`, `<base>_blue`) and either (a) `subst R: \path\to\base_red` etc. on Windows, or (b) on Linux export `COLIBRI_RED`, `COLIBRI_GREEN`, `COLIBRI_BLUE` and patch the lookup table that maps `TELESCOPE` → peer paths in `pipeline_automation.py`.

**Fixture night**

Generate the smallest plausible night that exercises all stages:
- 1 dark subdirectory with 9–11 dark frames
- 2–3 minute directories with ~30 frames each (well below the real 2400 — most stages don't care about count, only structure)
- 1 ACP log with at least one "Field Name:", "Sunset JD", "Sunrise JD", "Dome closed!" line
- A minimal `kernels.txt` for the secondary pipeline

The .rcd files can be either copies of one real frame replicated and renamed, or zero-filled dummies if a stage tolerates them. Document which stages need real photons (probably `colibri_main_py3` + `sensitivity`) and which only need the file to exist.

**Decision needed:** Windows-with-`subst` vs. Linux-with-symlinks. The telescope computers are Windows, and several scripts shell out to WSL (`wsl time solve-field …` in `astrometrynet_funcs.getLocalSolution`). A Linux dev environment will not run those branches verbatim. Recommendation: develop on Linux for orchestrator + idempotency work (Phase 1), and keep a Windows VM available to test the WSL-bound scripts before deploying.

---

### Phase 1 — Minimum viable robustness (do second; ~3–5 days)

Goal: replace the sentinel-file scheme with something that records *what ran, whether it succeeded, with what arguments, and when*, in a single per-night source of truth. Make all cross-telescope waits bounded. **Do not touch the science scripts in this phase** — only `pipeline_automation.py` and the small helpers around it.

**Concrete changes**

1. **Per-night status manifest.** Replace the scatter of `*.txt` sentinels with one JSON file per night:
   `D:/ColibriArchive/<YYYY-MM-DD>/pipeline_status.json`

   Schema (one entry per stage):
   ```json
   {
     "obsdate": "20260417",
     "telescope": "GREENBIRD",
     "stages": {
       "colibri_main_py3": {
         "status": "success",
         "exit_code": 0,
         "started_at": "2026-04-18T01:32:11Z",
         "finished_at": "2026-04-18T02:14:02Z",
         "args": ["d:/", "2026/04/17", "-s 4"],
         "log_path": "D:/Logs/Pipeline/20260417/colibri_main_py3.log",
         "error": null,
         "code_version": "<git-sha>"
       }
     }
   }
   ```
   Allowed `status` values: `pending`, `running`, `success`, `failed`, `timeout`, `skipped`.

2. **Atomic writes.** Always write to `pipeline_status.json.tmp` and `os.replace()` over the real file, so a crash mid-write cannot corrupt the manifest. Read with a try/except for malformed JSON and fall back to "treat all as pending."

3. **Honest exit-code handling.** Wrap `subprocess.run` (drop the `shell=True` while you're there — it's actively harmful with the `'|', 'tee'` argument list construction). Capture `returncode`, set status accordingly. Capture stderr to the log file and put the last ~20 lines into the manifest's `error` field.

4. **`repro` becomes meaningful.** Define:
   - default behavior: rerun any stage whose last `status` is not `success`
   - `--repro all`: rerun everything
   - `--repro <stage1>,<stage2>`: rerun specific stages
   - `--repro failed`: only rerun stages with `status == "failed"` or `"timeout"`

5. **Bounded cross-telescope waits.** Replace
   ```python
   while not (path_RED.is_file() == path_RED.parent.is_dir()):
       time.sleep(300)
   ```
   with
   ```python
   def wait_for_peer(peer_status_path, stage, timeout_s=6*3600, poll_s=120):
       deadline = time.monotonic() + timeout_s
       while time.monotonic() < deadline:
           try:
               peer = json.loads(peer_status_path.read_text())
               s = peer["stages"].get(stage, {}).get("status")
               if s == "success":
                   return True
               if s in ("failed", "timeout"):
                   return False
           except (FileNotFoundError, json.JSONDecodeError):
               pass
           time.sleep(poll_s)
       return False  # timed out
   ```
   On a `False` return, mark the local stage as `skipped` with reason `"peer_not_ready"` and *continue with whatever downstream work is independent of that peer*. Do not silently hang.

6. **Read peer status from the manifest, not from `done.txt`.** The manifest is the single source of truth; `done.txt` and `timeline_ready.txt` go away. Peers know whether `colibri_main_py3`, `wcsmatching`, etc. succeeded by reading the peer's `pipeline_status.json` over the existing `R:/`, `G:/`, `B:/` mounts.

7. **Backwards-compatible shadow mode (one or two nights).** Keep writing the legacy `*.txt` sentinels in addition to the manifest, but read decisions from the manifest. This lets you roll back instantly if something goes wrong.

8. **Delete `processdata.py`.** It is no longer the live entry point and its presence creates ambiguity. Remove the commented-out reference from `ProcessColibriData.bat`.

9. **Fix the `is_file() == is_dir()` idiom** wherever it appears, even in places we're not otherwise touching.

10. **Per-stage timeouts.** Each stage gets a configurable maximum wall-clock time (sensible defaults: primary pipeline = 6 h, all others = 1 h). On timeout, send SIGTERM, wait 30 s, then SIGKILL, and mark `status: "timeout"`.

**What is intentionally not done in Phase 1**
- No DAG scheduler — keep the explicit ordered dictionary in `ColibriProcesses` for now.
- No changes to the science scripts themselves, including the duplicate `*_pos.npy`-based skip logic in `colibri_main_py3.py`. (We accept the redundancy; the manifest is the authority for the orchestrator, the `.npy` check stays as the in-script optimization.)
- No removal of hardcoded `D:/` paths from the science scripts. Phase 0's simulator handles this for testing; Phase 2 fixes it properly.
- No new logging framework — keep `print` for now.

**Acceptance criteria for Phase 1**
- A simulated night with one stage forced to crash does not stall the pipeline; the failed stage is recorded and downstream stages that depend on it are explicitly skipped, while independent stages still run.
- Killing one of the three telescope orchestrators (e.g. `kill -9` on the simulator) results in the other two timing out their wait and producing a complete-but-degraded night within `wait_timeout` seconds.
- Re-running with default flags after a partial failure picks up exactly the failed/skipped stages and nothing else.

---

### Phase 2 — Production-quality refactor (do third; weeks, not days)

Once Phase 1 is in production for a few weeks and we trust the manifest, take on the structural cleanup:

1. **Centralized configuration.** A single `colibri.toml` (or YAML) that defines all paths, sigma thresholds, frame counts, peer telescope hostnames, drive letters, timeouts, kernel paths. Loaded once at startup, passed explicitly into every function. No more `BASE_PATH = pathlib.Path('D:/')` at module scope.

2. **Logging.** Replace `print` everywhere with the `logging` module. Per-night log directory, structured (key=value or JSON-line) format so it's grep-able. Stop redirecting via `'|', 'tee'`-style argument trickery.

3. **DAG scheduler.** Move from the hand-coded ordered dict in `ColibriProcesses` to a real DAG. Two reasonable options:
   - **Snakemake** — heavy in bioinformatics, increasingly used in astronomy, handles input/output dependencies, parallelism, retries, and provenance natively. Steepest learning curve but the most you-get-for-free.
   - **A small in-house DAG** — ~200 lines of Python: stages declare `inputs`, `outputs`, `depends_on`; the runner walks the graph, decides which to run based on the manifest + file mtimes, and parallelizes independent branches. Best if Snakemake is overkill or doesn't fit the telescope sync model.

4. **Cross-telescope sync.** The shared-mount + JSON manifest from Phase 1 is fine for a 3-node system. If the array grows, consider a tiny status server (a single Flask/FastAPI process on Green) or even Redis. Don't over-engineer this until needed.

5. **Tests.** Phase 0's simulator becomes a CI fixture. Each science script gets at least a smoke test (runs without crashing on the synthetic night) and ideally one numerical test (e.g. `colibri_main_py3` finds the planted dip in a synthetic light curve). The sensitivity-detection thresholds and kernel-matching numerical outputs are well-suited to regression tests with stored expected values.

6. **Containerize the per-telescope environment.** Three telescopes × three independent installs is a recipe for drift. A container image (or at minimum a pinned `requirements.txt` + `environment.yml`) eliminates "it works on Red but not on Blue" forever.

7. **Path handling.** Remove all string-concatenated paths and Windows-only separators. Pathlib everywhere. Remove the `'/D:'` and `'/D:/'` patterns that exist in some scripts.

8. **Delete dead code.** `biasPlots.py`, `readnoise.py` (per the README's own admission), `processdata.py`, `data_clean.py` if not in use, and any unreferenced helpers. Use `vulture` or equivalent to find them.

9. **Document the science.** Each top-level script gets a docstring describing the algorithm, its inputs, its outputs, and the relevant equations / paper references (Pass et al. for the sigma normalization, etc.). The current docstrings are sparse.

10. **Provenance.** Every output file (npy, det_*.txt, medstacked.fits) should record the git SHA of the pipeline that produced it, the input manifest, and the wall-clock time. This is non-negotiable for any work that ends up in a paper.

---

## 2. Open decisions (please pin down before Phase 1 starts)

1. **Manifest granularity.** Per-night-per-telescope (one file per `<YYYY-MM-DD>` per machine) is what's described above. Alternative: per-night-global (a single file replicated to all three machines). Per-machine is simpler and avoids write contention; per-global makes the cross-telescope reads trivial. Recommendation: per-machine.

2. **Manifest location.** `ColibriArchive/<YYYY-MM-DD>/pipeline_status.json` is convenient because it's already on the shared mount. Alternative: a separate `PipelineStatus/` tree. Recommendation: keep it in `ColibriArchive` so it travels with the data.

3. **Shadow mode duration.** How many nights do we run the new scheme alongside the old before deleting the sentinel-file logic? Recommendation: 5 successful nights minimum, 2 weeks ideal.

4. **Simulator OS.** Linux dev environment is faster and friendlier, but cannot exercise the WSL/astrometry path or the Windows network-mount semantics. Recommendation: Linux primary, Windows VM for final pre-deploy validation.

5. **Default wait timeout for cross-telescope handoffs.** 6 hours is enough to cover an entire long winter night; 1 hour is enough for any single stage. Recommendation: 8 h for the night-final wait, 1 h for intra-night waits.

6. **What to do when a peer reports `failed`.** Strict (refuse to run anything that would have used its output) vs. lenient (run what we can, mark the rest as `skipped: peer_failed`). Recommendation: lenient, with a prominent flag in the status email.

---

## 3. Risks and known unknowns

- **Synchronized rollout.** All three telescopes must run compatible versions of the orchestrator during the transition. If Green is updated and Red is not, Green's manifest reader must tolerate Red still using sentinels. The shadow-mode period covers this.
- **`generate_specific_lightcurve.txt` waiting loop.** Same unbounded-wait pattern as `done.txt`. Phase 1 fixes this too.
- **Timezones and the YYYYMMDD obsdate.** Verify which calendar day the night belongs to (sunset-local vs. UTC). Misidentifying this could cause the orchestrator to look in the wrong directory. The `getDataTimes` and `hyphonateDate` helpers should be unit-tested against this.
- **WCS / astrometry.net rate limits and outages.** Several stages depend on either the local WSL `solve-field` install or the astrometry.net web API. A timeout should not cause cascading failures — the stage should mark `failed` with a clear reason and downstream stages should degrade gracefully.
- **Disk-fill scenarios.** None of the cleanup logic checks whether `D:/` is full before writing. Out of scope for Phase 1, but worth a `shutil.disk_usage` check at the start of each night in Phase 2.
- **Cython modules (`colibri_image_reader.pyx`).** The `Init/` directory contains build scripts. The simulator must rebuild these for the dev OS. Document the build incantation in Phase 0.

---

## 4. Concrete next steps (this week)

1. Stand up the simulator directory tree on the work computer (Phase 0). Get `colibri_main_py3.py` to run end-to-end against it without writing to any real `D:/`.
2. Capture the current production behavior: run the live `pipeline_automation.py` against the simulator and record exactly which sentinels it produces, in which order, under (a) success, (b) one stage failing, (c) one peer never appearing. This is the baseline against which Phase 1 changes will be compared.
3. Pin down the open decisions in §2.
4. Write the manifest reader/writer module (~100 lines, no dependencies beyond stdlib) and unit-test it.
5. Begin Phase 1 surgery on `pipeline_automation.py`.

---

*Last updated: 2026-04-17. Update this document when assumptions change.*
