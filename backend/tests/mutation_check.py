"""Mutation check: break guarded behaviour on purpose, confirm the named test fails, always restore the file.

Run from backend/:  python tests/mutation_check.py     (every mutation should print CAUGHT)
The files are restored byte-for-byte in a finally block; check `git status` afterwards if a run was interrupted.
"""
import subprocess, sys
from pathlib import Path

B = Path(__file__).resolve().parents[1]      # backend/
PY = sys.executable

ALL = [
    ("idempotent detect removed", "app/api/routes/detect.py",
     "    if existing is not None:\n        return existing\n", "    if False:\n        return existing\n",
     "tests/test_detect_api.py::test_detection_is_idempotent"),
    ("provenance ownership check removed", "app/api/routes/detect.py",
     '''    upload = r.scalar_one_or_none()
    if not upload or upload.user_id != current_user.user_id:
        raise HTTPException(404, "No detection result found for this upload.")
    r2 = await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))
    result = r2.scalar_one_or_none()
    if not result:
        raise HTTPException(404, "No detection result found for this upload.")

    pv = await''',
     '''    upload = r.scalar_one_or_none()
    if not upload:
        raise HTTPException(404, "No detection result found for this upload.")
    r2 = await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))
    result = r2.scalar_one_or_none()
    if not result:
        raise HTTPException(404, "No detection result found for this upload.")

    pv = await''',
     "tests/test_upload_and_access.py"),
    ("video v1 stale labels back", "app/services/report/generator.py",
     'if "Video Model v1" in (result.model_used or ""):', 'if False:',
     "tests/test_report.py::test_video_v1_report_uses_the_correct_labels_not_the_old_heuristic_ones"),
    ("image threshold changed", "app/core/config.py", "FAKE_THRESHOLD: float = 0.5", "FAKE_THRESHOLD: float = 0.6",
     "tests/test_verdicts.py::test_default_threshold_is_unchanged"),
    ("explainability failure breaks report", "app/services/report/generator.py",
     '''    try:
        elements.extend(_explainability_flowables(upload, result, styles))
    except Exception:''',
     '''    try:
        elements.extend(_explainability_flowables(upload, result, styles))
        raise RuntimeError("x") if False else None
    except ZeroDivisionError:''',
     "tests/test_report.py::test_report_still_builds_when_the_explanation_cannot_be_computed"),
    ("tamper detection weakened", "app/services/provenance/c2pa_reader.py",
     "content_intact=(", "content_intact=True or (",
     "tests/test_provenance.py::test_tampering_after_signing_is_detected"),
    ("conflict note always shown", "app/services/provenance/service.py",
     'if verdict == "REAL":', 'if True:',
     "tests/test_provenance.py::test_conflict_note_only_when_the_verdict_disagrees"),
    ("evidence not faded", "app/services/explain/core.py",
     "return float(min(1.0, max(0.0, fake_probability / 0.5)))", "return 1.0",
     "tests/test_explain_core.py"),
    ("AUC ties mishandled", "eval/metrics.py",
     "ranks[order[i:j + 1]] = (i + j) / 2 + 1", "ranks[order[i:j + 1]] = i + 1",
     "tests/test_eval_metrics.py"),
    ("video scan blocks the event loop again", "app/services/video/detector.py",
     "result = await run_in_threadpool(backend.score, path, progress)", "result = backend.score(path, progress)",
     "tests/test_jobs.py::test_a_running_scan_does_not_block_other_requests"),
    ("cancel ignored while running", "app/services/jobs.py",
     '''            if job.cancel_requested:
                raise JobCancelled()
            with self._lock:
                job.progress''',
     '''            with self._lock:
                job.progress''',
     "tests/test_jobs.py::test_cancelling_a_running_job_stops_it_and_stores_nothing"),
    ("progress may go backwards", "app/services/jobs.py",
     "job.progress = max(job.progress, min(max(float(fraction), 0.0), 0.99))", "job.progress = float(fraction)",
     "tests/test_jobs.py::test_progress_never_decreases_and_stays_below_one_until_the_job_is_done"),
    ("duplicate active jobs allowed", "app/services/jobs.py",
     '''        if active_id and self._jobs.get(active_id) and self._jobs[active_id].state in ACTIVE:
            return self._jobs[active_id]
''',
     "",
     "tests/test_jobs.py::test_starting_twice_returns_the_same_job_and_never_duplicates_the_result"),
    ("job ownership check removed", "app/api/routes/jobs.py",
     "if job is None or job.user_id != user.user_id:", "if job is None:",
     "tests/test_jobs.py::test_jobs_are_private_to_their_owner"),
    ("feedback export ignores consent", "app/services/feedback_export.py",
     "if fb.allow_reuse and label is not None and Path(up.storage_url).is_file():",
     "if label is not None and Path(up.storage_url).is_file():",
     "tests/test_feedback.py::test_export_only_copies_files_and_comments_of_users_who_opted_in"),
    ("production start-up accepts the default secret", "app/core/startup.py",
     "if secret == DEFAULT_SECRET:", "if False:",
     "tests/test_deploy_helpers.py::test_the_app_itself_refuses_to_start_unsafely"),
    ("colab cache-fix leakage counter disabled", "eval/video_head_experiments/v5_cache_fix.py",
     'leak += sum(1 for k in c["keys"] if k in vk)', "leak += 0",
     "tests/test_v5_cache_fix.py::test_cache_report_counts_and_flags_leakage"),
]

MUTATIONS = ALL
caught = 0
for name, rel, old, new, target in MUTATIONS:
    path = B / rel
    original_bytes = path.read_bytes()          # bytes: restore must be exact (line endings included)
    original = original_bytes.decode("utf-8").replace("\r\n", "\n")   # match on LF; the exact bytes are restored below
    if old not in original:
        print(f"[SKIP ] {name}: pattern not found in {rel}")
        continue
    try:
        path.write_bytes(original.replace(old, new, 1).encode("utf-8"))
        r = subprocess.run([PY, "-m", "pytest", target, "-q", "-x", "-p", "no:cacheprovider"], cwd=B,
                           capture_output=True, text=True)
        failed = r.returncode != 0
    finally:
        path.write_bytes(original_bytes)
    caught += failed
    print(f"[{'CAUGHT' if failed else 'MISSED'}] {name}")
print(f"\n{caught}/{len(MUTATIONS)} mutations caught")

