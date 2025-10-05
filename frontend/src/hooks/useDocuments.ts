import {useCallback, useEffect, useState} from 'react'
import {useUploadApi} from '../services/uploadApi'
// Import Document type to ensure consistency
import { Document } from '../types'
import { useApi } from '../lib/api'
import { useAuth } from '@clerk/nextjs'



export const useDocuments = () => {
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retryCount, setRetryCount] = useState(0)
  const { isLoaded, isSignedIn } = useAuth()
  const api = useApi()

  // Real GET /api/documents
  const fetchDocuments = useCallback(async () => {
    // Don't fetch if Clerk is still loading or user is not signed in
    if (!isLoaded || !isSignedIn) {
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const fetchedDocuments = await api.getDocuments()
      setDocuments(fetchedDocuments)
      setRetryCount(0) // Reset retry count on success
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch documents'
      setError(errorMessage)
      setRetryCount(prev => {
        const newCount = prev + 1
        console.error('Error fetching documents (attempt', newCount, '):', err)
        return newCount
      })
    } finally {
      setIsLoading(false)
    }
  }, [api, isLoaded, isSignedIn])

  // Real-time status polling for processing documents
  useEffect(() => {
    if (!isLoaded || !isSignedIn) return

    const processingDocs = documents.filter(doc => doc.status === 'processing')

    if (processingDocs.length > 0) {
      const pollInterval = setInterval(async () => {
        try {
          // Fetch fresh document data from API to get current status
          const updatedDocuments = await api.getDocuments()
          setDocuments(updatedDocuments)
        } catch (err) {
          console.error('Error polling document status:', err)
          // Don't update error state here to avoid disrupting the UI during polling
        }
      }, 3000) // Poll every 3 seconds

      return () => clearInterval(pollInterval)
    }
  }, [documents, api, isLoaded, isSignedIn])

  // Initial fetch - wait for Clerk to load and user to be signed in
  useEffect(() => {
    if (isLoaded && isSignedIn) {
      fetchDocuments()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoaded, isSignedIn])

  return {
    documents,
    isLoading,
    error,
    refetch: fetchDocuments,
  }
}

export const useDocumentUpload = () => {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const { uploadDocument, isSignedIn } = useUploadApi()

  const upload = useCallback(async (file: File): Promise<Document> => {
    if (!isSignedIn) {
      throw new Error('You must be signed in to upload documents')
    }

    setIsUploading(true)
    setUploadProgress(0)
    setError(null)

    try {
        return await uploadDocument(file, (progress) => {
          setUploadProgress(progress)
      })
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed'
      setError(errorMessage)
      throw err
    } finally {
      setIsUploading(false)
      setUploadProgress(0)
    }
  }, [uploadDocument, isSignedIn])

  return {
    uploadDocument: upload,
    isUploading,
    uploadProgress,
    error,
  }
}

export const useDocumentDelete = () => {
  const [isDeleting, setIsDeleting] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const api = useApi()

  // Real DELETE /api/documents/{id}
  const deleteDocument = useCallback(async (id: number): Promise<void> => {
    setIsDeleting(id)
    setError(null)

    try {
      await api.deleteDocument(id)
      // Success - document deleted successfully
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete document'
      setError(errorMessage)
      console.error('Error deleting document:', err)
      throw err
    } finally {
      setIsDeleting(null)
    }
  }, [api])

  return {
    deleteDocument,
    isDeleting,
    error,
  }
}
