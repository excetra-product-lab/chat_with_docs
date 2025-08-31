'use client'

// API service for backend communication with Clerk authentication
import { useAuth } from '@clerk/nextjs'
import { ErrorHandler } from './errorHandler'
import { getEnvironmentConfig } from './env-validation'

// Get validated environment configuration
const envConfig = getEnvironmentConfig()
const API_URL = envConfig.NEXT_PUBLIC_API_URL

// Hook for authenticated API calls
export function useApi() {
  const { getToken, isSignedIn } = useAuth()

  // Get authentication headers with Clerk Bearer token
  const getAuthHeaders = async (): Promise<Record<string, string>> => {
    if (!isSignedIn) {
      throw new Error('User must be signed in to make API calls')
    }
    
    try {
      const token = await getToken()
      
      if (!token) {
        throw new Error('No authentication token available')
      }
      
      return {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    } catch (error) {
      console.error('Failed to get auth headers:', error)
      throw new Error('Authentication failed')
    }
  }

  // Authenticated API client with error handling
  const apiRequest = async (endpoint: string, options: RequestInit = {}) => {
    const url = `${API_URL}${endpoint}`
    
    try {
      // Get authentication headers
      const authHeaders = await getAuthHeaders()
      
      const response = await fetch(url, {
        headers: {
          ...authHeaders,
          ...options.headers, // Allow override of headers if needed
        },
        ...options,
      })

      if (!response.ok) {
        // Use our error handler for API errors
        throw await ErrorHandler.processApiError(response, { endpoint, method: options.method })
      }

      return response.json()
    } catch (error) {
      // If it's already an AppError from processApiError, re-throw it
      if (error && typeof error === 'object' && 'type' in error) {
        throw error
      }
      
      // Process other errors (network, etc.)
      throw ErrorHandler.processAuthError(error instanceof Error ? error : new Error('API request failed'), { endpoint })
    }
  }

  return {
    // Document endpoints
    uploadDocument: async (file: File) => {
      try {
        const formData = new FormData()
        formData.append('file', file)

        // Get auth headers but exclude Content-Type for FormData
        const authHeaders = await getAuthHeaders()
        const { 'Content-Type': _, ...authHeadersWithoutContentType } = authHeaders

        const response = await fetch(`${API_URL}/api/documents/upload`, {
          method: 'POST',
          headers: {
            ...authHeadersWithoutContentType, // Include auth but not Content-Type for FormData
          },
          body: formData, // Don't set Content-Type header for FormData
        })

        if (!response.ok) {
          throw await ErrorHandler.processApiError(response, { operation: 'upload', fileName: file.name })
        }

        return response.json()
      } catch (error) {
        // Use specific upload error handling
        throw ErrorHandler.processUploadError(
          error instanceof Error ? error : new Error('Upload failed'), 
          file.name,
          { fileSize: file.size, fileType: file.type }
        )
      }
    },

    getDocuments: async () => {
      return apiRequest('/api/documents')
    },

    deleteDocument: async (id: number) => {
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
    
    // Helper properties
    isSignedIn,
  }
}

// Legacy export for backward compatibility (will be updated in components)
export const api = {
  uploadDocument: async (file: File) => {
    throw new Error('Use useApi() hook instead of direct api calls')
  },
  getDocuments: async () => {
    throw new Error('Use useApi() hook instead of direct api calls')
  },
  deleteDocument: async (id: number) => {
    throw new Error('Use useApi() hook instead of direct api calls')
  },
  sendQuery: async (question: string, documentIds?: string[]) => {
    throw new Error('Use useApi() hook instead of direct api calls')
  },
}

export default api 