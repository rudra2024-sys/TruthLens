/**
 * Shared between MonitoringSession.jsx, ExamAttempt.jsx, and useMonitoringSession.js's guidance
 * text (added 2026-10-15) so all three always agree on the full set of reasons the backend can
 * actually flag -- mirrors backend/app/services/report/session_report.py's REASON_LABELS/
 * EVENT_LABELS exactly. Keeping one copy here is what caught the original bug: the hook's
 * student-facing guidance text had drifted to cover only 5 of these 11 reasons, so 6 kinds of
 * flag (deepfake_signal, deepfake_uncertain, looking_away, talking_detected, object_detected,
 * device_mismatch_suspected) produced a flag with no visible explanation at all.
 */

export const REASON_LABELS = {
  no_face: 'No face detected',
  multiple_faces: 'Multiple faces detected',
  identity_mismatch: 'Identity mismatch',
  identity_uncertain: 'Identity uncertain',
  deepfake_signal: 'Possible AI manipulation',
  deepfake_uncertain: 'Manipulation uncertain',
  looking_away: 'Looking away',
  talking_detected: 'Talking detected',
  object_detected: 'Unauthorized object detected',
  device_mismatch_suspected: 'Possible device change',
  static_image_suspected: 'Possible static photo detected',
}

export const EVENT_LABELS = {
  tab_hidden: 'Tab hidden',
  window_blurred: 'Window lost focus',
  clipboard_paste: 'Clipboard paste',
  devtools_suspected: 'Devtools suspected',
  camera_interrupted: 'Camera interrupted',
  live_no_face: 'Live: no face detected',
  live_looking_away: 'Live: looking away',
  live_multiple_faces: 'Live: second person detected',
}
