import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
})

function friendlyError(err) {
  // Added 2026-10-03: a 500's body carries a request_id (see backend/app/main.py's exception handler) so a
  // user-reported error is actually traceable in server logs - without this it's shown then lost forever.
  const requestId = err.response?.data?.request_id || err.response?.headers?.['x-request-id']
  const suffix = requestId ? ` (reference: ${requestId})` : ''

  if (err.response?.data?.detail) {
    const detail = err.response.data.detail

    if (Array.isArray(detail)) {
      return detail.map((x) => x.msg).join(', ')
    }

    return detail + suffix
  }

  if (err.code === 'ECONNABORTED') {
    return 'Request timed out. The server may be busy — try again.'
  }

  if (err.message === 'Network Error') {
    return 'Cannot reach the TruthLens backend. Is it running?'
  }

  return 'An unexpected error occurred.'
}

/*
 * Automatically attach JWT to every API request.
 */
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('truthlens_token')

    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }

    return config
  },
  (error) => Promise.reject(error)
)

/*
 * Automatically clear invalid authentication, and send the visitor back to /login so an
 * expired/invalid session surfaces as one clear message instead of a confusing 401 wherever it
 * happened to occur (added 2026-10-03: a mid-scan 401 used to clear the token silently, and the
 * *next* request in the same flow - e.g. a background-job poll - then went out with no
 * Authorization header at all, surfacing FastAPI's generic "Not authenticated" deep inside an
 * unrelated error state instead of a clear "your session expired" message where it's actionable).
 *
 * Excludes /auth/login and /auth/signup themselves: a 401 there means "wrong password", not "your
 * session expired" - redirecting would blow away the error message those pages already show, and
 * there's no session to lose in the first place.
 */
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const url = error.config?.url || ''
    const isAuthAttempt = url.includes('/auth/login') || url.includes('/auth/signup')

    if (error.response?.status === 401 && !isAuthAttempt) {
      const hadSession = !!localStorage.getItem('truthlens_token')
      localStorage.removeItem('truthlens_token')
      localStorage.removeItem('truthlens_user')

      const onAuthPage = window.location.pathname.startsWith('/login') || window.location.pathname.startsWith('/signup')
      if (hadSession && !onAuthPage) {
        window.location.href = '/login?expired=1'
      }
    }

    return Promise.reject(error)
  }
)

export const uploadMedia = async (file, onProgress) => {
  const form = new FormData()
  form.append('file', file)

  try {
    return await api.post('/upload/', form, {
      onUploadProgress: (e) => {
        if (e.total) {
          onProgress?.(
            Math.round((e.loaded * 100) / e.total)
          )
        }
      },
    })
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const runDetection = async (uploadId) => {
  try {
    return await api.post(`/detect/${uploadId}`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getHistory = async (limit = 20) => {
  try {
    return await api.get(`/history?limit=${limit}`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getStats = async () => {
  try {
    return await api.get('/stats')
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getDetectionResult = async (uploadId) => {
  try {
    return await api.get(`/detect/${uploadId}/result`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Feedback on a result ("was this correct?"). One per upload; PUT replaces it, DELETE withdraws it (and any consent to
// keep the file). GET resolves to null when nothing has been submitted.
export const getFeedback = async (uploadId) => {
  try {
    return (await api.get(`/detect/${uploadId}/feedback`)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const saveFeedback = async (uploadId, body) => {
  try {
    return (await api.put(`/detect/${uploadId}/feedback`, body)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const withdrawFeedback = async (uploadId) => {
  try {
    await api.delete(`/detect/${uploadId}/feedback`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Background detection jobs (used for video, which is slow): start returns immediately with a job id, then poll for
// progress. The finished result is fetched with getDetectionResult(uploadId).
export const startDetectionJob = async (uploadId) => {
  try {
    return await api.post(`/detect/${uploadId}/jobs`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getJob = async (jobId) => {
  try {
    return await api.get(`/jobs/${jobId}`)
  } catch (err) {
    const e = new Error(friendlyError(err))
    e.status = err?.response?.status
    throw e
  }
}

export const cancelJob = async (jobId) => {
  try {
    return await api.delete(`/jobs/${jobId}`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Explainability: Grad-CAM heatmap (image) or per-frame scores + face crops (video). Computed on demand by
// the backend, so it is only requested when the user opens the panel.
export const getExplanation = async (uploadId) => {
  try {
    return await api.get(`/detect/${uploadId}/explain`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Provenance: C2PA Content Credentials, embedded metadata and an ELA visual aid. Cheap (no model), read-only,
// supplementary to the verdict.
export const getProvenance = async (uploadId) => {
  try {
    return await api.get(`/detect/${uploadId}/provenance`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Identity-match: enroll a reference photo, then compare a later live-captured frame against
// it. GET resolves to null when nothing has been enrolled yet, same convention as feedback.
export const enrollIdentityReference = async (uploadId) => {
  try {
    return (await api.post('/identity/reference', { upload_id: uploadId })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getIdentityReference = async () => {
  try {
    return (await api.get('/identity/reference')).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const deleteIdentityReference = async () => {
  try {
    await api.delete('/identity/reference')
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const matchIdentity = async (uploadId) => {
  try {
    return (await api.post('/identity/match', { upload_id: uploadId })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Continuous monitoring: periodic identity/deepfake/presence checks within a session, any of
// which can flag independently -- including a check that flags even when identity matches.
export const startMonitoringSession = async () => {
  try {
    return (await api.post('/identity/sessions')).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const submitMonitoringCheck = async (sessionId, uploadId) => {
  try {
    return (await api.post(`/identity/sessions/${sessionId}/checks`, { upload_id: uploadId })).data
  } catch (err) {
    const e = new Error(friendlyError(err))
    e.status = err?.response?.status
    throw e
  }
}

export const endMonitoringSession = async (sessionId) => {
  try {
    return (await api.post(`/identity/sessions/${sessionId}/end`)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getMonitoringSession = async (sessionId) => {
  try {
    return (await api.get(`/identity/sessions/${sessionId}`)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Behavioral events (tab/window focus, clipboard, devtools heuristic, camera interruption) --
// no camera frame or model involved, posted directly as they're detected client-side.
export const postMonitoringEvent = async (sessionId, eventType, detail) => {
  try {
    return (await api.post(`/identity/sessions/${sessionId}/events`, { event_type: eventType, detail })).data
  } catch (err) {
    const e = new Error(friendlyError(err))
    e.status = err?.response?.status
    throw e
  }
}

export const downloadSessionReport = async (sessionId) => {
  try {
    const response = await api.get(`/identity/sessions/${sessionId}/report`, {
      responseType: 'blob',
    })

    const disposition = response.headers['content-disposition'] || ''
    const match = disposition.match(/filename="?([^";]+)"?/)
    const filename = match ? match[1] : `truthlens-session-report-${sessionId}.pdf`

    const blobUrl = window.URL.createObjectURL(response.data)
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(blobUrl)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getUpload = async (uploadId) => {
  try {
    return await api.get(`/upload/${uploadId}`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const reportUrl = (uploadId) =>
  `/api/v1/report/${uploadId}`

export const downloadReport = async (uploadId) => {
  try {
    const response = await api.get(`/report/${uploadId}`, {
      responseType: 'blob',
    })

    const disposition = response.headers['content-disposition'] || ''
    const match = disposition.match(/filename="?([^";]+)"?/)
    const filename = match ? match[1] : `truthlens-report-${uploadId}.pdf`

    const blobUrl = window.URL.createObjectURL(response.data)
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(blobUrl)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const signup = async (name, email, password) => {
  try {
    return await api.post('/auth/signup', {
      name,
      email,
      password,
    })
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const login = async (email, password) => {
  try {
    return await api.post('/auth/login', {
      email,
      password,
    })
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getCurrentUser = async () => {
  try {
    return await api.get('/auth/me')
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Permanently deletes the account and everything belonging to it (uploads, reports, results, feedback) --
// see backend/app/services/account_deletion.py. Irreversible; the caller is responsible for confirming with
// the user before calling this (see components/DeleteAccountModal.jsx) and logging out afterward.
export const deleteAccount = async () => {
  try {
    return await api.delete('/auth/me')
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const saveAuth = (authData) => {
  localStorage.setItem(
    'truthlens_token',
    authData.access_token
  )

  localStorage.setItem(
    'truthlens_user',
    JSON.stringify({
      user_id: authData.user_id,
      name: authData.name,
      email: authData.email,
    })
  )
}

export const logout = () => {
  localStorage.removeItem('truthlens_token')
  localStorage.removeItem('truthlens_user')
}

export const getStoredUser = () => {
  const stored = localStorage.getItem('truthlens_user')

  if (!stored) return null

  try {
    return JSON.parse(stored)
  } catch {
    localStorage.removeItem('truthlens_user')
    return null
  }
}

export default api