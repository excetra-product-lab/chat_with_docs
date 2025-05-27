import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_URL,
})

const api = {
  // Auth endpoints
  signup: async (email: string, password: string) => {
    return apiClient.post('/api/auth/signup', { email, password })
  },

  // Document endpoints
  uploadDocument: async (file: File, token: string) => {
    const formData = new FormData()
    formData.append('file', file)

    return apiClient.post('/api/documents/upload', formData, {
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'multipart/form-data',
      },
    })
  },

  getDocuments: async (token: string) => {
    return apiClient.get('/api/documents/', {
      headers: { Authorization: `Bearer ${token}` },
    })
  },

  deleteDocument: async (id: number, token: string) => {
    return apiClient.delete(`/api/documents/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
  },

  // Chat endpoints
  sendQuery: async (question: string, token: string) => {
    return apiClient.post(
      '/api/chat/query',
      { question },
      {
        headers: { Authorization: `Bearer ${token}` },
      }
    )
  },
}

export { api }
export default api
