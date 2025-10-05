import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook } from '@testing-library/react'
import {
  ErrorType,
  ErrorSeverity,
  categorizeError,
  getUserMessage,
  getRecoveryAction,
  getErrorSeverity,
  processError,
  ErrorHandler,
  useErrorHandler,
  type AppError
} from '../lib/errorHandler'

describe('errorHandler', () => {
  const originalNodeEnv = process.env.NODE_ENV

  beforeEach(() => {
    vi.clearAllMocks()
    // Mock console methods to avoid noise in tests
    vi.spyOn(console, 'group').mockImplementation(() => {})
    vi.spyOn(console, 'error').mockImplementation(() => {})
    vi.spyOn(console, 'groupEnd').mockImplementation(() => {})
  })

  afterEach(() => {
    (process.env as any).NODE_ENV = originalNodeEnv
    vi.restoreAllMocks()
  })

  describe('categorizeError', () => {
    it('should categorize network errors', () => {
      const networkErrors = [
        new Error('Network error occurred'),
        new Error('Fetch failed'),
        new Error('Connection timeout'),
        new Error('Connection refused'),
      ]

      networkErrors.forEach(error => {
        expect(categorizeError(error)).toBe(ErrorType.NETWORK)
      })
    })

    it('should categorize authentication errors', () => {
      const authErrors = [
        new Error('Unauthorized access'),
        new Error('Forbidden resource'),
        new Error('Invalid token'),
        new Error('Authentication failed'),
        new Error('HTTP 401: Unauthorized'),
        new Error('HTTP 403: Forbidden'),
      ]

      authErrors.forEach(error => {
        expect(categorizeError(error)).toBe(ErrorType.AUTHENTICATION)
      })
    })

    it('should categorize validation errors', () => {
      const validationErrors = [
        new Error('Validation failed'),
        new Error('Invalid input'),
        new Error('Required field missing'),
        new Error('HTTP 400: Bad Request'),
      ]

      validationErrors.forEach(error => {
        expect(categorizeError(error)).toBe(ErrorType.VALIDATION)
      })
    })

    it('should categorize server errors', () => {
      const serverErrors = [
        new Error('HTTP 500: Internal Server Error'),
        new Error('HTTP 502: Bad Gateway'),
        new Error('HTTP 503: Service Unavailable'),
        new Error('Server error occurred'),
        new Error('Internal error'),
      ]

      serverErrors.forEach(error => {
        expect(categorizeError(error)).toBe(ErrorType.SERVER)
      })
    })

    it('should categorize unknown errors', () => {
      const unknownErrors = [
        new Error('Something weird happened'),
        'String error',
        null,
        undefined,
        { message: 'Object error' },
      ]

      unknownErrors.forEach(error => {
        expect(categorizeError(error)).toBe(ErrorType.UNKNOWN)
      })
    })

    it('should handle case insensitive matching', () => {
      expect(categorizeError(new Error('NETWORK ERROR'))).toBe(ErrorType.NETWORK)
      expect(categorizeError(new Error('Network Error'))).toBe(ErrorType.NETWORK)
      expect(categorizeError(new Error('network error'))).toBe(ErrorType.NETWORK)
    })
  })

  describe('getUserMessage', () => {
    it('should return appropriate messages for each error type', () => {
      expect(getUserMessage(ErrorType.NETWORK)).toBe(
        'Connection problem. Please check your internet connection and try again.'
      )
      expect(getUserMessage(ErrorType.AUTHENTICATION)).toBe(
        'Authentication required. Please sign in to continue.'
      )
      expect(getUserMessage(ErrorType.VALIDATION)).toBe(
        'Please check your input and try again.'
      )
      expect(getUserMessage(ErrorType.SERVER)).toBe(
        'Server is experiencing issues. Please try again in a few moments.'
      )
      expect(getUserMessage(ErrorType.UNKNOWN)).toBe(
        'Something went wrong. Please try again.'
      )
    })

    it('should use original message for unknown errors when provided', () => {
      const originalMessage = 'Custom error message'
      expect(getUserMessage(ErrorType.UNKNOWN, originalMessage)).toBe(originalMessage)
    })

    it('should fallback to default message when original message is empty', () => {
      expect(getUserMessage(ErrorType.UNKNOWN, '')).toBe(
        'Something went wrong. Please try again.'
      )
    })
  })

  describe('getRecoveryAction', () => {
    it('should return appropriate recovery actions for each error type', () => {
      expect(getRecoveryAction(ErrorType.NETWORK)).toBe(
        'Check your connection and refresh the page'
      )
      expect(getRecoveryAction(ErrorType.AUTHENTICATION)).toBe('Sign in again')
      expect(getRecoveryAction(ErrorType.VALIDATION)).toBe(
        'Review your input and try again'
      )
      expect(getRecoveryAction(ErrorType.SERVER)).toBe('Wait a moment and try again')
      expect(getRecoveryAction(ErrorType.UNKNOWN)).toBe(
        'Refresh the page and try again'
      )
    })
  })

  describe('getErrorSeverity', () => {
    it('should return appropriate severity levels for each error type', () => {
      expect(getErrorSeverity(ErrorType.AUTHENTICATION)).toBe(ErrorSeverity.HIGH)
      expect(getErrorSeverity(ErrorType.SERVER)).toBe(ErrorSeverity.HIGH)
      expect(getErrorSeverity(ErrorType.NETWORK)).toBe(ErrorSeverity.MEDIUM)
      expect(getErrorSeverity(ErrorType.VALIDATION)).toBe(ErrorSeverity.LOW)
      expect(getErrorSeverity(ErrorType.UNKNOWN)).toBe(ErrorSeverity.MEDIUM)
    })
  })

  describe('processError', () => {
    it('should create a complete AppError object', () => {
      const originalError = new Error('Test error')
      const context = { operation: 'test', userId: '123' }
      
      const appError = processError(originalError, context)

      expect(appError).toEqual({
        type: ErrorType.UNKNOWN,
        severity: ErrorSeverity.MEDIUM,
        message: 'Test error',
        userMessage: 'Test error',
        recoveryAction: 'Refresh the page and try again',
        originalError,
        timestamp: expect.any(Date),
        context,
      })
    })

    it('should handle non-Error objects', () => {
      const appError = processError('String error')

      expect(appError.message).toBe('Unknown error occurred')
      expect(appError.originalError).toBeUndefined()
      expect(appError.type).toBe(ErrorType.UNKNOWN)
    })

    it('should log errors in development environment', () => {
      (process.env as any).NODE_ENV = 'development'
      
      const error = new Error('Test error')
      processError(error)

      expect(console.group).toHaveBeenCalledWith('🚨 UNKNOWN ERROR (medium)')
      expect(console.error).toHaveBeenCalledTimes(4)
      expect(console.groupEnd).toHaveBeenCalled()
    })

    it('should not log errors in production environment', () => {
      (process.env as any).NODE_ENV = 'production'
      
      const error = new Error('Test error')
      processError(error)

      expect(console.group).not.toHaveBeenCalled()
      expect(console.error).not.toHaveBeenCalled()
      expect(console.groupEnd).not.toHaveBeenCalled()
    })

    it('should set timestamp correctly', () => {
      const beforeTime = new Date()
      const appError = processError(new Error('Test'))
      const afterTime = new Date()

      expect(appError.timestamp.getTime()).toBeGreaterThanOrEqual(beforeTime.getTime())
      expect(appError.timestamp.getTime()).toBeLessThanOrEqual(afterTime.getTime())
    })
  })

  describe('ErrorHandler.processApiError', () => {
    it('should process API error with JSON detail', async () => {
      const mockResponse = new Response(
        JSON.stringify({ detail: 'Custom API error' }),
        {
          status: 400,
          statusText: 'Bad Request',
          headers: { 'Content-Type': 'application/json' }
        }
      )

      const appError = await ErrorHandler.processApiError(mockResponse, { endpoint: '/api/test' })

      expect(appError.message).toBe('Custom API error')
      expect(appError.context).toEqual({
        endpoint: '/api/test',
        status: 400,
        url: ''
      })
      // The error type depends on the error message content
      expect([ErrorType.VALIDATION, ErrorType.UNKNOWN]).toContain(appError.type)
    })

    it('should process API error with JSON message', async () => {
      const mockResponse = new Response(
        JSON.stringify({ message: 'Server error message' }),
        {
          status: 500,
          statusText: 'Internal Server Error',
          headers: { 'Content-Type': 'application/json' }
        }
      )

      const appError = await ErrorHandler.processApiError(mockResponse)

      expect(appError.message).toBe('Server error message')
      expect(appError.type).toBe(ErrorType.SERVER)
    })

    it('should fallback to status text when JSON parsing fails', async () => {
      const mockResponse = new Response(
        'Invalid JSON',
        {
          status: 404,
          statusText: 'Not Found',
          headers: { 'Content-Type': 'text/plain' }
        }
      )

      const appError = await ErrorHandler.processApiError(mockResponse)

      expect(appError.message).toBe('HTTP 404: Not Found')
    })

    it('should handle response without detail or message', async () => {
      const mockResponse = new Response(
        JSON.stringify({ other: 'field' }),
        {
          status: 422,
          statusText: 'Unprocessable Entity',
          headers: { 'Content-Type': 'application/json' }
        }
      )

      const appError = await ErrorHandler.processApiError(mockResponse)

      expect(appError.message).toBe('HTTP 422: Unprocessable Entity')
    })
  })

  describe('ErrorHandler.processAuthError', () => {
    it('should process authentication error with context', () => {
      const error = new Error('Token expired')
      const context = { userId: '123' }

      const appError = ErrorHandler.processAuthError(error, context)

      expect(appError.message).toBe('Token expired')
      expect(appError.type).toBe(ErrorType.AUTHENTICATION)
      expect(appError.context).toEqual({
        userId: '123',
        component: 'clerk-auth'
      })
    })
  })

  describe('ErrorHandler.processUploadError', () => {
    it('should process upload error with file context', () => {
      const error = new Error('File too large')
      const fileName = 'document.pdf'
      const context = { fileSize: 1024000 }

      const appError = ErrorHandler.processUploadError(error, fileName, context)

      expect(appError.message).toBe('File too large')
      expect(appError.context).toEqual({
        fileSize: 1024000,
        fileName: 'document.pdf',
        operation: 'file-upload'
      })
    })
  })

  describe('ErrorHandler.processChatError', () => {
    it('should process chat error with query context', () => {
      const error = new Error('RAG processing failed')
      const query = 'What is this document about?'
      const context = { documentIds: ['1', '2'] }

      const appError = ErrorHandler.processChatError(error, query, context)

      expect(appError.message).toBe('RAG processing failed')
      expect(appError.context).toEqual({
        documentIds: ['1', '2'],
        query: 'What is this document about?',
        operation: 'chat-query'
      })
    })
  })

  describe('useErrorHandler', () => {
    it('should provide error handling functions', () => {
      const { result } = renderHook(() => useErrorHandler())

      expect(typeof result.current.handleError).toBe('function')
      expect(typeof result.current.handleApiError).toBe('function')
      expect(typeof result.current.processAuthError).toBe('function')
      expect(typeof result.current.processUploadError).toBe('function')
      expect(typeof result.current.processChatError).toBe('function')
    })

    it('should handle error correctly', () => {
      const { result } = renderHook(() => useErrorHandler())
      const error = new Error('Test error')
      const context = { operation: 'test' }

      const appError = result.current.handleError(error, context)

      expect(appError.message).toBe('Test error')
      expect(appError.context).toEqual(context)
    })

    it('should handle API error correctly', async () => {
      const { result } = renderHook(() => useErrorHandler())
      const mockResponse = new Response(
        JSON.stringify({ detail: 'API error' }),
        { status: 400, statusText: 'Bad Request' }
      )

      const appError = await result.current.handleApiError(mockResponse)

      expect(appError.message).toBe('API error')
      // The error type depends on the error message content
      expect([ErrorType.VALIDATION, ErrorType.UNKNOWN]).toContain(appError.type)
    })

    it('should reference static methods correctly', () => {
      const { result } = renderHook(() => useErrorHandler())

      expect(result.current.processAuthError).toBe(ErrorHandler.processAuthError)
      expect(result.current.processUploadError).toBe(ErrorHandler.processUploadError)
      expect(result.current.processChatError).toBe(ErrorHandler.processChatError)
    })
  })

  describe('Error type and severity enums', () => {
    it('should have correct error types', () => {
      expect(ErrorType.NETWORK).toBe('network')
      expect(ErrorType.AUTHENTICATION).toBe('authentication')
      expect(ErrorType.VALIDATION).toBe('validation')
      expect(ErrorType.SERVER).toBe('server')
      expect(ErrorType.UNKNOWN).toBe('unknown')
    })

    it('should have correct severity levels', () => {
      expect(ErrorSeverity.LOW).toBe('low')
      expect(ErrorSeverity.MEDIUM).toBe('medium')
      expect(ErrorSeverity.HIGH).toBe('high')
      expect(ErrorSeverity.CRITICAL).toBe('critical')
    })
  })
})
