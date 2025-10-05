'use client'

import { useAuth } from '@clerk/nextjs'
import type { Document } from '../types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

export function useUploadApi() {
  const { getToken, isSignedIn, isLoaded } = useAuth()

  const uploadDocument = async (
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<Document> => {
    if (!isLoaded) {
      throw new Error('Authentication is still loading, please wait')
    }
    
    if (!isSignedIn) {
      throw new Error('You must be signed in to upload documents')
    }

    const url = `${API_BASE_URL}/api/documents/upload`

    return new Promise(async (resolve, reject) => {
      try {
        const token = await getToken()
        const xhr = new XMLHttpRequest()

        // Handle upload progress
        if (onProgress) {
          xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
              const progress = Math.round((event.loaded / event.total) * 100)
              onProgress(progress)
            }
          })
        }

        // Handle response
        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const response = JSON.parse(xhr.responseText)
              resolve(response)
            } catch (error) {
              reject(new Error('Failed to parse response'))
            }
          } else {
            try {
              const errorData = JSON.parse(xhr.responseText)
              reject(new Error(errorData.detail || `HTTP ${xhr.status}: ${xhr.statusText}`))
            } catch {
              reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`))
            }
          }
        })

        // Handle errors
        xhr.addEventListener('error', () => {
          reject(new Error('Network error occurred'))
        })

        // Prepare form data
        const formData = new FormData()
        formData.append('file', file)

        // Set up request
        xhr.open('POST', url)

        // Add auth header if we have a token
        if (token) {
          xhr.setRequestHeader('Authorization', `Bearer ${token}`)
        }

        // Send request
        xhr.send(formData)
      } catch (error) {
        reject(error)
      }
    })
  }

  return {
    uploadDocument,
    isSignedIn,
  }
}
