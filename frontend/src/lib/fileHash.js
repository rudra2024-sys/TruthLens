// SHA-256 of a File's bytes as lowercase hex, or null when the browser can't do it (no Web Crypto outside a secure
// context, unreadable file). Callers treat null as "can't tell, just scan it".
export async function sha256Hex(file) {
  try {
    if (!globalThis.crypto?.subtle) return null
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
  } catch {
    return null
  }
}
