/**
 * Single source of truth for which file types TruthLens actually accepts.
 * Mirrors backend/app/core/config.py's ALLOWED_IMAGE_TYPES / ALLOWED_VIDEO_TYPES /
 * ALLOWED_AUDIO_TYPES exactly — update both together if the backend's accepted
 * types ever change. Previously Home.jsx and Verify.jsx each hardcoded their own
 * (different, both incomplete) format lists; this is the one place that decides
 * what the <input accept> attribute and the displayed format chips both show.
 */
export const ACCEPTED_MIME_TYPES = [
  'image/jpeg', 'image/png', 'image/webp',
  'video/mp4', 'video/avi', 'video/quicktime', 'video/webm',
  'audio/wav', 'audio/mpeg', 'audio/flac', 'audio/x-wav', 'audio/mp3',
  'audio/mp4', 'audio/x-m4a', 'audio/m4a',
]

export const ACCEPTED_INPUT_ACCEPT = ACCEPTED_MIME_TYPES.join(',')

export const ACCEPTED_FORMAT_CHIPS = ['JPG', 'PNG', 'WEBP', 'MP4', 'MOV', 'AVI', 'WEBM', 'WAV', 'MP3', 'FLAC', 'M4A']
