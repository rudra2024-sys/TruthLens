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
export const enrollIdentityReference = async (uploadId, consent) => {
  try {
    return (await api.post('/identity/reference', { upload_id: uploadId, consent })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Transient-use government-ID verification (added 2026-10-12) -- the uploaded ID photo is
// deleted server-side immediately after use, success or failure; only the resulting face
// embedding is kept. See identity.py::post_verify_government_id.
export const verifyGovernmentId = async (uploadId, consent) => {
  try {
    return (await api.post('/identity/verify-government-id', { upload_id: uploadId, consent })).data
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

// Organisation linking (added 2026-10-12) -- a student joins via a private join code before
// enrollment is allowed; GET resolves to null when not yet linked, same null-means-unset
// convention as getIdentityReference above.
export const joinOrganization = async (joinCode) => {
  try {
    return (await api.post('/identity/organization', { join_code: joinCode })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getMyOrganization = async () => {
  try {
    return (await api.get('/identity/organization')).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getOrganizationJoinCode = async () => {
  try {
    return (await api.get('/admin/join-code')).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const regenerateOrganizationJoinCode = async () => {
  try {
    return (await api.post('/admin/join-code/regenerate')).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// The current user's own past monitoring sessions (added 2026-10-12).
export const getMySessions = async () => {
  try {
    return (await api.get('/identity/sessions')).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

// Proctored MCQ exams (added 2026-10-12) -- org-authored, taken inside TruthLens itself with
// continuous monitoring. See backend/app/api/routes/exams.py.
export const createExam = async (title, passThreshold, maxFlags) => {
  try {
    return (await api.post('/exams', { title, pass_threshold: passThreshold, max_flags: maxFlags })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getMyExams = async () => {
  try {
    return (await api.get('/exams')).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const updateExam = async (examId, title, passThreshold, maxFlags) => {
  try {
    return (await api.patch(`/exams/${examId}`, { title, pass_threshold: passThreshold, max_flags: maxFlags })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const addExamQuestion = async (examId, text, options, correctOptionIndex) => {
  try {
    return (await api.post(`/exams/${examId}/questions`, {
      text, options, correct_option_index: correctOptionIndex,
    })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getExamQuestionsAdmin = async (examId) => {
  try {
    return (await api.get(`/exams/${examId}/questions/admin`)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const deleteExamQuestion = async (examId, questionId) => {
  try {
    await api.delete(`/exams/${examId}/questions/${questionId}`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getExam = async (examId) => {
  try {
    return (await api.get(`/exams/${examId}`)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getExamQuestions = async (examId) => {
  try {
    return (await api.get(`/exams/${examId}/questions`)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const startExamAttempt = async (examId, details = {}) => {
  try {
    return (await api.post(`/exams/${examId}/attempts`, { details })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getExamAttempt = async (attemptId) => {
  try {
    return (await api.get(`/exams/attempts/${attemptId}`)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const submitExamAnswer = async (attemptId, questionId, selectedOptionIndex) => {
  try {
    return (await api.post(`/exams/attempts/${attemptId}/answers`, {
      question_id: questionId, selected_option_index: selectedOptionIndex,
    })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const finishExamAttempt = async (attemptId) => {
  try {
    return (await api.post(`/exams/attempts/${attemptId}/finish`)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getExamAttemptsAdmin = async (examId) => {
  try {
    return (await api.get(`/admin/exams/${examId}/attempts`)).data
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

export const submitMonitoringCheck = async (sessionId, uploadId, audioUploadId) => {
  try {
    return (await api.post(`/identity/sessions/${sessionId}/checks`, {
      upload_id: uploadId,
      audio_upload_id: audioUploadId,
    })).data
  } catch (err) {
    const e = new Error(friendlyError(err))
    e.status = err?.response?.status
    throw e
  }
}

// Background check jobs (added 2026-10-11, item 6) -- same start/poll shape as
// startDetectionJob/getJob above, sized for a monitoring session's per-tick check instead of a
// one-off video scan. The finished check is embedded in the job itself (job.result), no separate
// fetch needed.
export const startMonitoringCheckJob = async (sessionId, uploadId, audioUploadId) => {
  try {
    return (await api.post(`/identity/sessions/${sessionId}/checks/jobs`, {
      upload_id: uploadId,
      audio_upload_id: audioUploadId,
    })).data
  } catch (err) {
    const e = new Error(friendlyError(err))
    e.status = err?.response?.status
    throw e
  }
}

export const getMonitoringCheckJob = async (sessionId, jobId) => {
  try {
    return (await api.get(`/identity/sessions/${sessionId}/checks/jobs/${jobId}`)).data
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

// Admin/proctor dashboard (added 2026-10-11) -- every call here 403s unless the signed-in
// account has role "admin" (see backend/scripts/promote_admin.py, the only way to get it).
export const getAdminSessions = async ({ flaggedOnly = false, limit = 50, offset = 0 } = {}) => {
  try {
    return (await api.get('/admin/sessions', {
      params: { flagged_only: flaggedOnly, limit, offset },
    })).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const getAdminSessionDetail = async (sessionId) => {
  try {
    return (await api.get(`/admin/sessions/${sessionId}`)).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const downloadAdminSessionReport = async (sessionId) => {
  try {
    const response = await api.get(`/admin/sessions/${sessionId}/report`, { responseType: 'blob' })
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

// Organisation roster (added 2026-10-12) -- the students currently linked to this organisation,
// separate from the session list since a student with zero sessions yet still belongs here.
export const getRoster = async () => {
  try {
    return (await api.get('/admin/roster')).data
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const removeFromRoster = async (userId) => {
  try {
    await api.delete(`/admin/roster/${userId}`)
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

export const signup = async (name, email, password, accountType) => {
  try {
    return await api.post('/auth/signup', {
      name,
      email,
      password,
      account_type: accountType,
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
      role: authData.role,
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