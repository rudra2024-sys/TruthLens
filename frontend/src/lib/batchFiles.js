/**
 * Collecting files for a batch scan: from the file picker (many files, or a whole folder) or from files / folders
 * dropped onto the page. Only images, videos and audio the backend accepts are kept; everything else in a dropped
 * folder (documents, .DS_Store, ...) is skipped and counted, not an error.
 */
export const MAX_BATCH_FILES = 100

const EXT_KIND = {
  jpg: 'image', jpeg: 'image', png: 'image', webp: 'image', heic: 'image', heif: 'image',
  mp4: 'video', mov: 'video', avi: 'video', webm: 'video',
  wav: 'audio', mp3: 'audio', flac: 'audio', m4a: 'audio', opus: 'audio', ogg: 'audio', oga: 'audio',
}

export function kindFromName(name) {
  const m = /\.([a-z0-9]+)$/i.exec(name || '')
  return m ? EXT_KIND[m[1].toLowerCase()] || null : null
}

/** [{ file, path }] from an <input type=file> FileList (path keeps the folder structure for a folder pick). */
export function entriesFromFileList(list) {
  return Array.from(list || []).map((file) => ({ file, path: file.webkitRelativePath || file.name }))
}

const readAll = (reader) => new Promise((resolve, reject) => {
  const out = []
  const next = () => reader.readEntries((batch) => {
    if (!batch.length) return resolve(out)
    out.push(...batch)
    next()
  }, reject)
  next()
})

async function walk(entry, prefix, out) {
  if (entry.isFile) {
    const file = await new Promise((res, rej) => entry.file(res, rej))
    out.push({ file, path: prefix + entry.name })
  } else if (entry.isDirectory) {
    for (const child of await readAll(entry.createReader())) await walk(child, `${prefix}${entry.name}/`, out)
  }
}

/** [{ file, path }] from a drop event's DataTransfer, descending into dropped folders. */
export async function entriesFromDrop(dataTransfer) {
  const out = []
  const items = Array.from(dataTransfer?.items || [])
  const roots = items.map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null))
  if (roots.length && roots.every(Boolean)) {
    for (const root of roots) await walk(root, '', out)
    return out
  }
  return entriesFromFileList(dataTransfer?.files)
}

/**
 * Splits raw entries into what can be scanned and what cannot.
 * `existing` = how many files are already in the batch. Returns { accepted, skippedType, skippedOverLimit }.
 */
export function selectBatchFiles(entries, existing = 0, max = MAX_BATCH_FILES) {
  const accepted = []
  let skippedType = 0
  let skippedOverLimit = 0
  for (const e of entries) {
    const hidden = e.file.name.startsWith('.')
    const kind = kindFromName(e.file.name)
    if (hidden || !kind) { if (!hidden) skippedType++; continue }
    if (existing + accepted.length >= max) { skippedOverLimit++; continue }
    accepted.push({ ...e, kind })
  }
  return { accepted, skippedType, skippedOverLimit }
}
