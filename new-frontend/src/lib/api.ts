// API service for backend communication
// This will be used in Task 3 for document upload and management

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Simple fetch-based API client (no authentication for now)
const apiRequest = async (endpoint: string, options: RequestInit = {}) => {
  const url = `${API_URL}${endpoint}`
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`)
  }

  return response.json()
}

export const api = {
  // Document endpoints
  uploadDocument: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_URL}/api/documents/upload`, {
      method: 'POST',
      body: formData, // Don't set Content-Type header for FormData
    })

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status} ${response.statusText}`)
    }

    return response.json()
  },

  getDocuments: async () => {
    return apiRequest('/api/documents')
  },

  deleteDocument: async (id: string) => {
    return apiRequest(`/api/documents/${id}`, {
      method: 'DELETE',
    })
  },

  // Chat endpoints
  sendQuery: async (question: string, documentIds?: string[]) => {
    return apiRequest('/api/chat/query', {
      method: 'POST',
      body: JSON.stringify({ 
        question,
        document_ids: documentIds 
      }),
    })
  },
}

export default api 