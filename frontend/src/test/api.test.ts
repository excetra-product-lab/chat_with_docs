import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useApi } from '../lib/api'
import { 
  mockClerkAuth, 
  mockApiResponses, 
  createMockResponse, 
  createMockErrorResponse,
  waitForAsync 
} from './utils'

// Mock functions with vi.hoisted to avoid hoisting issues
const mockUseAuth = vi.hoisted(() => vi.fn())

// Mock Clerk
vi.mock('@clerk/nextjs', () => ({
  useAuth: mockUseAuth,
}))

// Mock environment validation
vi.mock('../lib/env-validation', () => ({
  getEnvironmentConfig: vi.fn(() => ({
    NEXT_PUBLIC_API_URL: 'http://localhost:8000',
  })),
}))

// Mock error handler
vi.mock('../lib/errorHandler', () => ({
  ErrorHandler: {
    processApiError: vi.fn((response) => {
      return Promise.resolve(new Error(`API Error: ${response.status}`))
    }),
    processAuthError: vi.fn((error) => error),
    processUploadError: vi.fn((error) => error),
  },
}))

describe('useApi', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('when user is signed in', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue(mockClerkAuth.signedIn)
    })

    describe('getDocuments', () => {
      it('should fetch documents successfully', async () => {
        const mockResponse = createMockResponse(mockApiResponses.documents.success)
        ;(global.fetch as any).mockResolvedValueOnce(mockResponse)

        const { result } = renderHook(() => useApi())
        const documents = await result.current.getDocuments()

        expect(global.fetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/documents',
          expect.objectContaining({
            headers: expect.objectContaining({
              'Authorization': 'Bearer mock-token-123',
              'Content-Type': 'application/json',
            }),
          })
        )
        expect(documents).toEqual(mockApiResponses.documents.success)
      })

      it('should handle API errors', async () => {
        const errorResponse = createMockErrorResponse(
          401,
          'Unauthorized',
          mockApiResponses.errors.unauthorized.body
        )
        ;(global.fetch as any).mockResolvedValueOnce(errorResponse)

        const { result } = renderHook(() => useApi())

        await expect(result.current.getDocuments()).rejects.toThrow('API Error: 401')
      })

      it('should handle network errors', async () => {
        ;(global.fetch as any).mockRejectedValueOnce(new Error('Network error'))

        const { result } = renderHook(() => useApi())

        await expect(result.current.getDocuments()).rejects.toThrow('Network error')
      })
    })

    describe('deleteDocument', () => {
      it('should delete document successfully', async () => {
        const mockResponse = createMockResponse({ success: true })
        ;(global.fetch as any).mockResolvedValueOnce(mockResponse)

        const { result } = renderHook(() => useApi())
        const response = await result.current.deleteDocument(1)

        expect(global.fetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/documents/1',
          expect.objectContaining({
            method: 'DELETE',
            headers: expect.objectContaining({
              'Authorization': 'Bearer mock-token-123',
              'Content-Type': 'application/json',
            }),
          })
        )
        expect(response).toEqual({ success: true })
      })

      it('should handle document not found error', async () => {
        const errorResponse = createMockErrorResponse(
          404,
          'Not Found',
          mockApiResponses.errors.notFound.body
        )
        ;(global.fetch as any).mockResolvedValueOnce(errorResponse)

        const { result } = renderHook(() => useApi())

        await expect(result.current.deleteDocument(999)).rejects.toThrow('API Error: 404')
      })
    })

    describe('sendQuery', () => {
      it('should send chat query successfully', async () => {
        const mockResponse = createMockResponse(mockApiResponses.chat.success)
        ;(global.fetch as any).mockResolvedValueOnce(mockResponse)

        const { result } = renderHook(() => useApi())
        const response = await result.current.sendQuery('What is this document about?')

        expect(global.fetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/chat/query',
          expect.objectContaining({
            method: 'POST',
            headers: expect.objectContaining({
              'Authorization': 'Bearer mock-token-123',
              'Content-Type': 'application/json',
            }),
            body: JSON.stringify({
              question: 'What is this document about?',
              document_ids: undefined,
            }),
          })
        )
        expect(response).toEqual(mockApiResponses.chat.success)
      })

      it('should send query with document IDs', async () => {
        const mockResponse = createMockResponse(mockApiResponses.chat.success)
        ;(global.fetch as any).mockResolvedValueOnce(mockResponse)

        const { result } = renderHook(() => useApi())
        await result.current.sendQuery('What is this document about?', ['1', '2'])

        expect(global.fetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/chat/query',
          expect.objectContaining({
            body: JSON.stringify({
              question: 'What is this document about?',
              document_ids: ['1', '2'],
            }),
          })
        )
      })

      it('should handle chat API errors', async () => {
        const errorResponse = createMockErrorResponse(
          500,
          'Internal Server Error',
          mockApiResponses.errors.serverError.body
        )
        ;(global.fetch as any).mockResolvedValueOnce(errorResponse)

        const { result } = renderHook(() => useApi())

        await expect(
          result.current.sendQuery('What is this document about?')
        ).rejects.toThrow('API Error: 500')
      })
    })

    describe('uploadDocument', () => {
      it('should upload document successfully', async () => {
        const mockResponse = createMockResponse(mockApiResponses.upload.success)
        ;(global.fetch as any).mockResolvedValueOnce(mockResponse)

        const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
        const { result } = renderHook(() => useApi())
        const response = await result.current.uploadDocument(mockFile)

        expect(global.fetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/documents/upload',
          expect.objectContaining({
            method: 'POST',
            headers: expect.objectContaining({
              'Authorization': 'Bearer mock-token-123',
              // Content-Type should not be set for FormData
            }),
            body: expect.any(FormData),
          })
        )

        // Verify FormData contains the file
        const call = (global.fetch as any).mock.calls[0]
        const formData = call[1].body as FormData
        expect(formData.get('file')).toBe(mockFile)

        expect(response).toEqual(mockApiResponses.upload.success)
      })

      it('should handle upload errors', async () => {
        const errorResponse = createMockErrorResponse(
          422,
          'Unprocessable Entity',
          mockApiResponses.errors.validationError.body
        )
        ;(global.fetch as any).mockResolvedValueOnce(errorResponse)

        const mockFile = new File(['test content'], 'test.txt', { type: 'text/plain' })
        const { result } = renderHook(() => useApi())

        await expect(result.current.uploadDocument(mockFile)).rejects.toThrow()
      })

      it('should not include Content-Type header for file uploads', async () => {
        const mockResponse = createMockResponse(mockApiResponses.upload.success)
        ;(global.fetch as any).mockResolvedValueOnce(mockResponse)

        const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
        const { result } = renderHook(() => useApi())
        await result.current.uploadDocument(mockFile)

        const call = (global.fetch as any).mock.calls[0]
        const headers = call[1].headers
        expect(headers['Content-Type']).toBeUndefined()
        expect(headers['Authorization']).toBe('Bearer mock-token-123')
      })
    })
  })

  describe('when user is not signed in', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue(mockClerkAuth.signedOut)
    })

    it('should throw authentication error for getDocuments', async () => {
      const { result } = renderHook(() => useApi())

      await expect(result.current.getDocuments()).rejects.toThrow(
        'User must be signed in to make API calls'
      )
    })

    it('should throw authentication error for deleteDocument', async () => {
      const { result } = renderHook(() => useApi())

      await expect(result.current.deleteDocument(1)).rejects.toThrow(
        'User must be signed in to make API calls'
      )
    })

    it('should throw authentication error for sendQuery', async () => {
      const { result } = renderHook(() => useApi())

      await expect(result.current.sendQuery('test')).rejects.toThrow(
        'User must be signed in to make API calls'
      )
    })

    it('should throw authentication error for uploadDocument', async () => {
      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const { result } = renderHook(() => useApi())

      await expect(result.current.uploadDocument(mockFile)).rejects.toThrow(
        'User must be signed in to make API calls'
      )
    })

    it('should return correct isSignedIn status', () => {
      const { result } = renderHook(() => useApi())
      expect(result.current.isSignedIn).toBe(false)
    })
  })

  describe('when authentication token is not available', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue({
        getToken: vi.fn().mockResolvedValue(null),
        isSignedIn: true,
        userId: 'user_123',
      })
    })

    it('should throw no token error', async () => {
      const { result } = renderHook(() => useApi())

      await expect(result.current.getDocuments()).rejects.toThrow(
        'Authentication failed'
      )
    })
  })

  describe('when token retrieval fails', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue(mockClerkAuth.expired)
    })

    it('should handle token retrieval errors', async () => {
      const { result } = renderHook(() => useApi())

      await expect(result.current.getDocuments()).rejects.toThrow(
        'Authentication failed'
      )
    })
  })

  describe('legacy api export', () => {
    it('should throw error for direct api usage', async () => {
      const { api } = await import('../lib/api')

      await expect(api.getDocuments()).rejects.toThrow(
        'Use useApi() hook instead of direct api calls'
      )
      await expect(api.deleteDocument(1)).rejects.toThrow(
        'Use useApi() hook instead of direct api calls'
      )
      await expect(api.sendQuery('test')).rejects.toThrow(
        'Use useApi() hook instead of direct api calls'
      )

      const mockFile = new File(['test'], 'test.pdf')
      await expect(api.uploadDocument(mockFile)).rejects.toThrow(
        'Use useApi() hook instead of direct api calls'
      )
    })
  })
})
