import axios from 'axios'

const api = axios.create({ baseURL: '/api/v1', timeout: 60000 })

function friendlyError(err) {
  if (err.response?.data?.detail) return err.response.data.detail
  if (err.code === 'ECONNABORTED') return 'Request timed out. The server may be busy — try again.'
  if (err.message === 'Network Error') {
    return 'Cannot reach the TruthLens backend. Is it running on http://localhost:8000 (local uvicorn or Docker)?'
  }
  return 'An unexpected error occurred.'
}

export const uploadMedia = async (file, onProgress) => {
  const form = new FormData()
  form.append('file', file)
  try {
    return await api.post('/upload/', form, {
      onUploadProgress: e => onProgress && onProgress(Math.round(e.loaded * 100 / e.total))
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

export const reportUrl = (uploadId) => `/api/v1/report/${uploadId}`
