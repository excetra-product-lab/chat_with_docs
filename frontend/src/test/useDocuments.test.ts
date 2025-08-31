import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { useDocuments, useDocumentUpload, useDocumentDelete } from '../hooks/useDocuments'
import { 
  mockClerkAuth, 
  mockApiResponses, 
  mockDocuments,
  createMockResponse,
  createMockErrorResponse,
  waitForAsync 
} from './utils'

// Mock functions with vi.hoisted to avoid hoisting issues
const mockUseApi = vi.hoisted(() => vi.fn())
const mockUseUploadApi = vi.hoisted(() => vi.fn())
const mockUseAuth = vi.hoisted(() => vi.fn())

// Mock dependencies
vi.mock('../lib/api', () => ({
  useApi: mockUseApi,
}))

vi.mock('../services/uploadApi', () => ({
  useUploadApi: mockUseUploadApi,
}))

vi.mock('@clerk/nextjs', () => ({
  useAuth: mockUseAuth,
}))

describe('useDocuments', () => {
  const mockApi = {
    getDocuments: vi.fn(),
    deleteDocument: vi.fn(),
    sendQuery: vi.fn(),
    uploadDocument: vi.fn(),
    isSignedIn: true,
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockUseApi.mockReturnValue(mockApi)
    // Clear timers
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('initial load', () => {
    it('should fetch documents on mount', async () => {
      mockApi.getDocuments.mockResolvedValue(mockDocuments)

      const { result } = renderHook(() => useDocuments())

      expect(result.current.isLoading).toBe(true)
      expect(result.current.documents).toEqual([])

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(mockApi.getDocuments).toHaveBeenCalledTimes(1)
      expect(result.current.documents).toEqual(mockDocuments)
      expect(result.current.error).toBe(null)
    })

    it('should handle fetch errors', async () => {
      const errorMessage = 'Failed to fetch documents'
      mockApi.getDocuments.mockRejectedValue(new Error(errorMessage))

      const { result } = renderHook(() => useDocuments())

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(result.current.error).toBe(errorMessage)
      expect(result.current.documents).toEqual([])
    })

    it('should handle non-Error objects', async () => {
      mockApi.getDocuments.mockRejectedValue('String error')

      const { result } = renderHook(() => useDocuments())

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(result.current.error).toBe('Failed to fetch documents')
    })
  })

  describe('refetch functionality', () => {
    it('should refetch documents when refetch is called', async () => {
      mockApi.getDocuments.mockResolvedValue(mockDocuments)

      const { result } = renderHook(() => useDocuments())

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      mockApi.getDocuments.mockClear()
      mockApi.getDocuments.mockResolvedValue([...mockDocuments, {
        id: 4,
        name: 'new-document.pdf',
        status: 'processed',
        size: 1024,
        uploaded_at: '2024-01-01T13:00:00Z',
        processed_at: '2024-01-01T13:01:00Z',
        chunk_count: 5,
        user_id: 'user_123',
      }])

      act(() => {
        result.current.refetch()
      })

      expect(result.current.isLoading).toBe(true)

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(mockApi.getDocuments).toHaveBeenCalledTimes(1)
      expect(result.current.documents).toHaveLength(4)
    })
  })

  describe('polling for processing documents', () => {
    it('should poll for processing documents', async () => {
      const processingDocument = {
        id: 1,
        name: 'processing.pdf',
        status: 'processing',
        size: 1024,
        uploaded_at: '2024-01-01T12:00:00Z',
        processed_at: null,
        chunk_count: 0,
        user_id: 'user_123',
      }

      // First call returns processing document
      mockApi.getDocuments.mockResolvedValueOnce([processingDocument])

      const { result } = renderHook(() => useDocuments())

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(result.current.documents).toEqual([processingDocument])

      // Mock the polling call - document is now processed
      const processedDocument = { ...processingDocument, status: 'processed', chunk_count: 10 }
      mockApi.getDocuments.mockResolvedValueOnce([processedDocument])

      // Fast-forward timer to trigger polling
      act(() => {
        vi.advanceTimersByTime(3000)
      })

      await waitFor(() => {
        expect(result.current.documents[0].status).toBe('processed')
      })

      expect(mockApi.getDocuments).toHaveBeenCalledTimes(2)
    })

    it('should stop polling when no documents are processing', async () => {
      // Start with processing document
      const processingDocument = {
        id: 1,
        name: 'processing.pdf',
        status: 'processing',
        size: 1024,
        uploaded_at: '2024-01-01T12:00:00Z',
        processed_at: null,
        chunk_count: 0,
        user_id: 'user_123',
      }

      mockApi.getDocuments.mockResolvedValueOnce([processingDocument])

      const { result } = renderHook(() => useDocuments())

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      // Document becomes processed
      const processedDocument = { ...processingDocument, status: 'processed', chunk_count: 10 }
      mockApi.getDocuments.mockResolvedValueOnce([processedDocument])

      act(() => {
        vi.advanceTimersByTime(3000)
      })

      await waitFor(() => {
        expect(result.current.documents[0].status).toBe('processed')
      })

      // Clear previous calls
      mockApi.getDocuments.mockClear()

      // Advance timer again - should not call API since no processing documents
      act(() => {
        vi.advanceTimersByTime(3000)
      })

      await waitForAsync(100)

      expect(mockApi.getDocuments).not.toHaveBeenCalled()
    })

    it('should handle polling errors gracefully', async () => {
      const processingDocument = {
        id: 1,
        name: 'processing.pdf',
        status: 'processing',
        size: 1024,
        uploaded_at: '2024-01-01T12:00:00Z',
        processed_at: null,
        chunk_count: 0,
        user_id: 'user_123',
      }

      mockApi.getDocuments.mockResolvedValueOnce([processingDocument])

      const { result } = renderHook(() => useDocuments())

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      // Mock polling error
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      mockApi.getDocuments.mockRejectedValueOnce(new Error('Polling failed'))

      act(() => {
        vi.advanceTimersByTime(3000)
      })

      await waitForAsync(100)

      expect(consoleErrorSpy).toHaveBeenCalledWith('Error polling document status:', expect.any(Error))
      // Error state should not be updated during polling
      expect(result.current.error).toBe(null)

      consoleErrorSpy.mockRestore()
    })
  })
})

describe('useDocumentUpload', () => {
  const mockUploadApi = {
    uploadDocument: vi.fn(),
    isSignedIn: true,
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockUseUploadApi.mockReturnValue(mockUploadApi)
    mockUseAuth.mockReturnValue(mockClerkAuth.signedIn)
  })

  it('should upload document successfully', async () => {
    const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
    const expectedResponse = mockApiResponses.upload.success

    mockUploadApi.uploadDocument.mockResolvedValue(expectedResponse)

    const { result } = renderHook(() => useDocumentUpload())

    expect(result.current.isUploading).toBe(false)
    expect(result.current.uploadProgress).toBe(0)
    expect(result.current.error).toBe(null)

    let uploadPromise: Promise<any>
    act(() => {
      uploadPromise = result.current.uploadDocument(mockFile)
    })

    expect(result.current.isUploading).toBe(true)

    const response = await uploadPromise!

    expect(mockUploadApi.uploadDocument).toHaveBeenCalledWith(
      mockFile,
      expect.any(Function)
    )
    expect(response).toEqual(expectedResponse)
    expect(result.current.isUploading).toBe(false)
    expect(result.current.uploadProgress).toBe(0)
    expect(result.current.error).toBe(null)
  })

  it('should track upload progress', async () => {
    const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })

    mockUploadApi.uploadDocument.mockImplementation((file, progressCallback) => {
      // Simulate progress updates
      setTimeout(() => progressCallback?.(25), 10)
      setTimeout(() => progressCallback?.(50), 20)
      setTimeout(() => progressCallback?.(75), 30)
      setTimeout(() => progressCallback?.(100), 40)
      return Promise.resolve(mockApiResponses.upload.success)
    })

    const { result } = renderHook(() => useDocumentUpload())

    act(() => {
      result.current.uploadDocument(mockFile)
    })

    await waitFor(() => {
      expect(result.current.uploadProgress).toBeGreaterThan(0)
    })

    await waitFor(() => {
      expect(result.current.isUploading).toBe(false)
    })

    expect(result.current.uploadProgress).toBe(0) // Reset after completion
  })

  it('should handle upload errors', async () => {
    const mockFile = new File(['test content'], 'test.txt', { type: 'text/plain' })
    const errorMessage = 'Invalid file format'

    mockUploadApi.uploadDocument.mockRejectedValue(new Error(errorMessage))

    const { result } = renderHook(() => useDocumentUpload())

    let uploadPromise: Promise<any>
    act(() => {
      uploadPromise = result.current.uploadDocument(mockFile)
    })

    await expect(uploadPromise!).rejects.toThrow(errorMessage)

    expect(result.current.error).toBe(errorMessage)
    expect(result.current.isUploading).toBe(false)
    expect(result.current.uploadProgress).toBe(0)
  })

  it('should handle non-Error objects', async () => {
    const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })

    mockUploadApi.uploadDocument.mockRejectedValue('String error')

    const { result } = renderHook(() => useDocumentUpload())

    let uploadPromise: Promise<any>
    act(() => {
      uploadPromise = result.current.uploadDocument(mockFile)
    })

    await expect(uploadPromise!).rejects.toThrow('String error')

    expect(result.current.error).toBe('Upload failed')
  })

      it('should throw error when user is not signed in', async () => {
      mockUseUploadApi.mockReturnValue({
        ...mockUploadApi,
        isSignedIn: false,
      })

    const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
    const { result } = renderHook(() => useDocumentUpload())

    await expect(result.current.uploadDocument(mockFile)).rejects.toThrow(
      'You must be signed in to upload documents'
    )
  })
})

describe('useDocumentDelete', () => {
  const mockApi = {
    getDocuments: vi.fn(),
    deleteDocument: vi.fn(),
    sendQuery: vi.fn(),
    uploadDocument: vi.fn(),
    isSignedIn: true,
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockUseApi.mockReturnValue(mockApi)
  })

  it('should delete document successfully', async () => {
    mockApi.deleteDocument.mockResolvedValue({ success: true })

    const { result } = renderHook(() => useDocumentDelete())

    expect(result.current.isDeleting).toBe(null)
    expect(result.current.error).toBe(null)

    let deletePromise: Promise<any>
    act(() => {
      deletePromise = result.current.deleteDocument(1)
    })

    expect(result.current.isDeleting).toBe(1)

    await deletePromise!

    expect(mockApi.deleteDocument).toHaveBeenCalledWith(1)
    expect(result.current.isDeleting).toBe(null)
    expect(result.current.error).toBe(null)
  })

  it('should handle delete errors', async () => {
    const errorMessage = 'Document not found'
    mockApi.deleteDocument.mockRejectedValue(new Error(errorMessage))

    const { result } = renderHook(() => useDocumentDelete())

    let deletePromise: Promise<any>
    act(() => {
      deletePromise = result.current.deleteDocument(999)
    })

    await expect(deletePromise!).rejects.toThrow(errorMessage)

    expect(result.current.error).toBe(errorMessage)
    expect(result.current.isDeleting).toBe(null)
  })

  it('should handle non-Error objects', async () => {
    mockApi.deleteDocument.mockRejectedValue('String error')

    const { result } = renderHook(() => useDocumentDelete())

    let deletePromise: Promise<any>
    act(() => {
      deletePromise = result.current.deleteDocument(1)
    })

    await expect(deletePromise!).rejects.toThrow('String error')

    expect(result.current.error).toBe('Failed to delete document')
  })

  it('should track which document is being deleted', async () => {
    // Simulate slow delete operation
    mockApi.deleteDocument.mockImplementation(() => 
      new Promise(resolve => setTimeout(() => resolve({ success: true }), 100))
    )

    const { result } = renderHook(() => useDocumentDelete())

    act(() => {
      result.current.deleteDocument(5)
    })

    expect(result.current.isDeleting).toBe(5)

    await waitFor(() => {
      expect(result.current.isDeleting).toBe(null)
    })
  })

  it('should handle multiple delete operations correctly', async () => {
    mockApi.deleteDocument.mockResolvedValue({ success: true })

    const { result } = renderHook(() => useDocumentDelete())

    // Start first delete
    act(() => {
      result.current.deleteDocument(1)
    })

    expect(result.current.isDeleting).toBe(1)

    await waitFor(() => {
      expect(result.current.isDeleting).toBe(null)
    })

    // Start second delete
    act(() => {
      result.current.deleteDocument(2)
    })

    expect(result.current.isDeleting).toBe(2)

    await waitFor(() => {
      expect(result.current.isDeleting).toBe(null)
    })

    expect(mockApi.deleteDocument).toHaveBeenCalledTimes(2)
    expect(mockApi.deleteDocument).toHaveBeenNthCalledWith(1, 1)
    expect(mockApi.deleteDocument).toHaveBeenNthCalledWith(2, 2)
  })
})
