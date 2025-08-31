import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useUploadApi } from '../services/uploadApi'
import { 
  mockClerkAuth, 
  mockApiResponses, 
  createMockXMLHttpRequest,
  waitForAsync 
} from './utils'

// Mock functions with vi.hoisted to avoid hoisting issues
const mockUseAuth = vi.hoisted(() => vi.fn())

// Mock Clerk
vi.mock('@clerk/nextjs', () => ({
  useAuth: mockUseAuth,
}))

describe('useUploadApi', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset XMLHttpRequest mock
    global.XMLHttpRequest = vi.fn() as any
  })

  describe('when user is signed in', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue(mockClerkAuth.signedIn)
    })

    it('should upload document successfully', async () => {
      const mockXHR = createMockXMLHttpRequest({
        shouldSucceed: true,
        responseData: mockApiResponses.upload.success,
      })
      ;(global.XMLHttpRequest as any).mockImplementation(() => mockXHR)

      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const { result } = renderHook(() => useUploadApi())

      const response = await result.current.uploadDocument(mockFile)

      expect(mockXHR.open).toHaveBeenCalledWith('POST', 'http://localhost:8000/api/documents/upload')
      expect(mockXHR.setRequestHeader).toHaveBeenCalledWith('Authorization', 'Bearer mock-token-123')
      expect(mockXHR.send).toHaveBeenCalledWith(expect.any(FormData))
      expect(response).toEqual(mockApiResponses.upload.success)
    })

    it('should track upload progress', async () => {
      const mockXHR = createMockXMLHttpRequest({
        shouldSucceed: true,
        shouldProgress: true,
        responseData: mockApiResponses.upload.success,
      })
      ;(global.XMLHttpRequest as any).mockImplementation(() => mockXHR)

      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const progressCallback = vi.fn()
      const { result } = renderHook(() => useUploadApi())

      await result.current.uploadDocument(mockFile, progressCallback)

      await waitFor(() => {
        expect(progressCallback).toHaveBeenCalled()
      })

      // Check that progress was reported
      const progressCalls = progressCallback.mock.calls
      expect(progressCalls.length).toBeGreaterThan(0)
      expect(progressCalls.some(call => call[0] === 50)).toBe(true)
      expect(progressCalls.some(call => call[0] === 100)).toBe(true)
    })

    it('should handle upload errors with valid JSON response', async () => {
      const mockXHR = createMockXMLHttpRequest({
        shouldSucceed: false,
        errorStatus: 422,
        errorMessage: 'Invalid file format',
      })
      ;(global.XMLHttpRequest as any).mockImplementation(() => mockXHR)

      const mockFile = new File(['test content'], 'test.txt', { type: 'text/plain' })
      const { result } = renderHook(() => useUploadApi())

      await expect(result.current.uploadDocument(mockFile)).rejects.toThrow('Invalid file format')
    })

    it('should handle upload errors with invalid JSON response', async () => {
      const mockXHR = createMockXMLHttpRequest({
        shouldSucceed: false,
        errorStatus: 500,
        errorMessage: 'Internal Server Error',
      })
      // Override responseText to invalid JSON
      mockXHR.responseText = 'Invalid JSON'
      ;(global.XMLHttpRequest as any).mockImplementation(() => mockXHR)

      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const { result } = renderHook(() => useUploadApi())

      await expect(result.current.uploadDocument(mockFile)).rejects.toThrow('HTTP 500: Internal Server Error')
    })

    it('should handle network errors', async () => {
      const mockXHR = {
        upload: {
          addEventListener: vi.fn(),
        },
        addEventListener: vi.fn((event: string, callback: Function) => {
          if (event === 'error') {
            setTimeout(() => callback(), 10)
          }
        }),
        open: vi.fn(),
        setRequestHeader: vi.fn(),
        send: vi.fn(),
      }
      ;(global.XMLHttpRequest as any).mockImplementation(() => mockXHR)

      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const { result } = renderHook(() => useUploadApi())

      await expect(result.current.uploadDocument(mockFile)).rejects.toThrow('Network error occurred')
    })

    it('should handle JSON parsing errors on success response', async () => {
      const mockXHR = createMockXMLHttpRequest({
        shouldSucceed: true,
        responseData: 'invalid json',
      })
      // Override responseText to invalid JSON
      mockXHR.responseText = 'invalid json'
      ;(global.XMLHttpRequest as any).mockImplementation(() => mockXHR)

      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const { result } = renderHook(() => useUploadApi())

      await expect(result.current.uploadDocument(mockFile)).rejects.toThrow('Failed to parse response')
    })

    it('should not call progress callback when not provided', async () => {
      const mockXHR = createMockXMLHttpRequest({
        shouldSucceed: true,
        shouldProgress: false,
        responseData: mockApiResponses.upload.success,
      })
      ;(global.XMLHttpRequest as any).mockImplementation(() => mockXHR)

      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const { result } = renderHook(() => useUploadApi())

      await result.current.uploadDocument(mockFile)

      // Should not set up progress listener if no callback provided
      expect(mockXHR.upload.addEventListener).not.toHaveBeenCalledWith(
        'progress',
        expect.any(Function)
      )
    })

    it('should handle progress events without lengthComputable', async () => {
      const mockXHR = {
        upload: {
          addEventListener: vi.fn((event: string, callback: Function) => {
            if (event === 'progress') {
              // Simulate progress event without lengthComputable
              setTimeout(() => callback({ lengthComputable: false, loaded: 50, total: 100 }), 10)
            }
          }),
        },
        addEventListener: vi.fn((event: string, callback: Function) => {
          if (event === 'load') {
            setTimeout(() => {
              mockXHR.status = 200
              mockXHR.statusText = 'OK'
              mockXHR.responseText = JSON.stringify(mockApiResponses.upload.success)
              callback()
            }, 20)
          }
        }),
        open: vi.fn(),
        setRequestHeader: vi.fn(),
        send: vi.fn(),
        status: 0,
        statusText: '',
        responseText: '',
      }
      ;(global.XMLHttpRequest as any).mockImplementation(() => mockXHR)

      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const progressCallback = vi.fn()
      const { result } = renderHook(() => useUploadApi())

      await result.current.uploadDocument(mockFile, progressCallback)

      // Progress callback should not be called for non-computable progress
      expect(progressCallback).not.toHaveBeenCalled()
    })

    it('should return correct isSignedIn status', () => {
      const { result } = renderHook(() => useUploadApi())
      expect(result.current.isSignedIn).toBe(true)
    })
  })

  describe('when user is not signed in', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue(mockClerkAuth.signedOut)
    })

    it('should throw authentication error', async () => {
      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const { result } = renderHook(() => useUploadApi())

      await expect(result.current.uploadDocument(mockFile)).rejects.toThrow(
        'You must be signed in to upload documents'
      )
    })

    it('should return correct isSignedIn status', () => {
      const { result } = renderHook(() => useUploadApi())
      expect(result.current.isSignedIn).toBe(false)
    })
  })

  describe('when token retrieval fails', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue(mockClerkAuth.expired)
    })

    it('should handle token retrieval errors', async () => {
      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const { result } = renderHook(() => useUploadApi())

      await expect(result.current.uploadDocument(mockFile)).rejects.toThrow('Token expired')
    })
  })

  describe('without authentication token', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue({
        getToken: vi.fn().mockResolvedValue(null),
        isSignedIn: true,
        userId: 'user_123',
      })
    })

    it('should upload without Authorization header when no token', async () => {
      const mockXHR = createMockXMLHttpRequest({
        shouldSucceed: true,
        responseData: mockApiResponses.upload.success,
      })
      ;(global.XMLHttpRequest as any).mockImplementation(() => mockXHR)

      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const { result } = renderHook(() => useUploadApi())

      await result.current.uploadDocument(mockFile)

      expect(mockXHR.setRequestHeader).not.toHaveBeenCalledWith(
        'Authorization',
        expect.any(String)
      )
    })
  })
})
