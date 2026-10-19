/**
 * Single source of truth for which file types TruthLens actually accepts.
 * Mirrors backend/app/core/config.py's ALLOWED_IMAGE_TYPES / ALLOWED_VIDEO_TYPES /
 * ALLOWED_AUDIO_TYPES exactly — update both together if the backend's accepted
 * types ever change. Previously Home.jsx and Verify.jsx each hardcoded their own
 * (different, both incomplete) format lists; this is the one place that decides
 * what the <input accept> attribute and the displayed format chips both show.
 */
export const ACCEPTED_MIME_TYPES = [
  'image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif',
  'video/mp4', 'video/avi', 'video/quicktime', 'video/webm',
  'audio/wav', 'audio/mpeg', 'audio/flac', 'audio/x-wav', 'audio/mp3',
  'audio/mp4', 'audio/x-m4a', 'audio/m4a',
  'audio/ogg', 'audio/opus',
]

// Browsers/OSes don't all map .opus/.heic/.heif to a MIME type, so the picker also lists these by
// extension (the backend falls back to the extension the same way, see upload.py::resolve_content_type).
export const ACCEPTED_EXTENSIONS = ['.opus', '.ogg', '.oga', '.heic', '.heif']

export const ACCEPTED_INPUT_ACCEPT = [...ACCEPTED_MIME_TYPES, ...ACCEPTED_EXTENSIONS].join(',')

export const ACCEPTED_FORMAT_CHIPS = ['JPG', 'PNG', 'WEBP', 'HEIC', 'MP4', 'MOV', 'AVI', 'WEBM', 'WAV', 'MP3', 'FLAC', 'M4A', 'OPUS']
