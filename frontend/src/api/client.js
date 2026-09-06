import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
})

function friendlyError(err) {
  if (err.response?.data?.detail) {
    const detail = err.response.data.detail

    if (Array.isArray(detail)) {
      return detail.map((x) => x.msg).join(', ')
    }

    return detail
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
 * Automatically clear invalid authentication.
 */
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('truthlens_token')
      localStorage.removeItem('truthlens_user')
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

export const getUpload = async (uploadId) => {
  try {
    return await api.get(`/upload/${uploadId}`)
  } catch (err) {
    throw new Error(friendlyError(err))
  }
}

export const reportUrl = (uploadId) =>
  `/api/v1/report/${uploadId}`

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