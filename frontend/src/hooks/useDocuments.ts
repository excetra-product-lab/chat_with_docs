import React, { useState, useEffect, useCallback } from 'react'

// Import Document type to ensure consistency
import { Document } from '../types'

// Mock API responses
const mockDocuments: Document[] = [
  {
    id: '1',
    filename: 'Contract_Analysis_2024.pdf',
    user_id: 'user1',
    status: 'ready',
    created_at: '2024-12-27T10:30:00Z',
    file_size: 2547829,
    file_type: 'application/pdf',
    pages: 12,
  },
  {
    id: '2',
    filename: 'Legal_Brief_Summary.docx',
    user_id: 'user1',
    status: 'processing',
    created_at: '2024-12-27T11:15:00Z',
    file_size: 1024000,
    file_type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    pages: 8,
    upload_progress: 75,
  },
  {
    id: '3',
    filename: 'Case_Notes.txt',
    user_id: 'user1',
    status: 'ready',
    created_at: '2024-12-27T09:45:00Z',
    file_size: 54000,
    file_type: 'text/plain',
  }
]

// Simulate API delays
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

export const useDocuments = () => {
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Mock GET /api/documents
  const fetchDocuments = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      await delay(500) // Simulate network delay
      setDocuments([...mockDocuments])
    } catch (err) {
      setError('Failed to fetch documents')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Status polling for processing documents
  useEffect(() => {
    const processingDocs = documents.filter(doc => doc.status === 'processing')

    if (processingDocs.length > 0) {
      const pollInterval = setInterval(async () => {
        // Mock status updates - randomly complete processing documents
        setDocuments(prev => prev.map(doc => {
          if (doc.status === 'processing' && Math.random() > 0.7) {
            return { ...doc, status: 'ready' as const, upload_progress: 100 }
          }
          if (doc.status === 'processing' && doc.upload_progress! < 100) {
            return { ...doc, upload_progress: Math.min(100, (doc.upload_progress || 0) + 10) }
          }
          return doc
        }))
      }, 2000) // Poll every 2 seconds

      return () => clearInterval(pollInterval)
    }
  }, [documents])

  // Initial fetch
  useEffect(() => {
    fetchDocuments()
  }, [fetchDocuments])

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

  // Mock POST /api/documents/upload
  const uploadDocument = useCallback(async (file: File): Promise<Document> => {
    setIsUploading(true)
    setUploadProgress(0)
    setError(null)

    try {
      // Simulate upload progress
      for (let progress = 0; progress <= 100; progress += 20) {
        await delay(200)
        setUploadProgress(progress)
      }

      await delay(500) // Final processing delay

      // Create mock uploaded document
      const newDocument: Document = {
        id: Date.now().toString(),
        filename: file.name,
        user_id: 'user1',
        status: 'processing',
        created_at: new Date().toISOString(),
        file_size: file.size,
        file_type: file.type,
        pages: Math.floor(file.size / 2000), // Mock page calculation
        upload_progress: 0,
      }

      return newDocument
    } catch (err) {
      setError('Upload failed')
      throw err
    } finally {
      setIsUploading(false)
      setUploadProgress(0)
    }
  }, [])

  return {
    uploadDocument,
    isUploading,
    uploadProgress,
    error,
  }
}

export const useDocumentDelete = () => {
  const [isDeleting, setIsDeleting] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Mock DELETE /api/documents/{id}
  const deleteDocument = useCallback(async (id: string): Promise<void> => {
    setIsDeleting(id)
    setError(null)

    try {
      await delay(800) // Simulate network delay
      // In real app, this would call the API
      // Mock success response
    } catch (err) {
      setError('Failed to delete document')
      throw err
    } finally {
      setIsDeleting(null)
    }
  }, [])

  return {
    deleteDocument,
    isDeleting,
    error,
  }
}
