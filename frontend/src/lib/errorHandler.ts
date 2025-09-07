// Error types for categorization
export enum ErrorType {
  NETWORK = 'network',
  AUTHENTICATION = 'authentication',
  VALIDATION = 'validation',
  SERVER = 'server',
  UNKNOWN = 'unknown'
}

// Error severity levels
export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// Structured error interface
export interface AppError {
  type: ErrorType
  severity: ErrorSeverity
  message: string
  userMessage: string
  recoveryAction?: string
  originalError?: Error
  timestamp: Date
  context?: Record<string, any>
}

// Error categorization function
export function categorizeError(error: Error | unknown): ErrorType {
  if (error instanceof Error) {
    const errorMessage = error.message.toLowerCase()
    
    // Network-related errors
    if (errorMessage.includes('network') || 
        errorMessage.includes('fetch') ||
        errorMessage.includes('connection') ||
        errorMessage.includes('timeout')) {
      return ErrorType.NETWORK
    }
    
    // Authentication errors
    if (errorMessage.includes('unauthorized') ||
        errorMessage.includes('forbidden') ||
        errorMessage.includes('token') ||
        errorMessage.includes('authentication') ||
        errorMessage.includes('401') ||
        errorMessage.includes('403')) {
      return ErrorType.AUTHENTICATION
    }
    
    // Validation errors
    if (errorMessage.includes('validation') ||
        errorMessage.includes('invalid') ||
        errorMessage.includes('required') ||
        errorMessage.includes('400')) {
      return ErrorType.VALIDATION
    }
    
    // Server errors
    if (errorMessage.includes('500') ||
        errorMessage.includes('502') ||
        errorMessage.includes('503') ||
        errorMessage.includes('server') ||
        errorMessage.includes('internal')) {
      return ErrorType.SERVER
    }
  }
  
  return ErrorType.UNKNOWN
}

// Get user-friendly message for error types
export function getUserMessage(errorType: ErrorType, originalMessage?: string): string {
  switch (errorType) {
    case ErrorType.NETWORK:
      return 'Connection problem. Please check your internet connection and try again.'
    case ErrorType.AUTHENTICATION:
      return 'Authentication required. Please sign in to continue.'
    case ErrorType.VALIDATION:
      return 'Please check your input and try again.'
    case ErrorType.SERVER:
      return 'Server is experiencing issues. Please try again in a few moments.'
    case ErrorType.UNKNOWN:
    default:
      return originalMessage || 'Something went wrong. Please try again.'
  }
}

// Get recovery action for error types
export function getRecoveryAction(errorType: ErrorType): string {
  switch (errorType) {
    case ErrorType.NETWORK:
      return 'Check your connection and refresh the page'
    case ErrorType.AUTHENTICATION:
      return 'Sign in again'
    case ErrorType.VALIDATION:
      return 'Review your input and try again'
    case ErrorType.SERVER:
      return 'Wait a moment and try again'
    case ErrorType.UNKNOWN:
    default:
      return 'Refresh the page and try again'
  }
}

// Get error severity based on type
export function getErrorSeverity(errorType: ErrorType): ErrorSeverity {
  switch (errorType) {
    case ErrorType.AUTHENTICATION:
      return ErrorSeverity.HIGH
    case ErrorType.SERVER:
      return ErrorSeverity.HIGH
    case ErrorType.NETWORK:
      return ErrorSeverity.MEDIUM
    case ErrorType.VALIDATION:
      return ErrorSeverity.LOW
    case ErrorType.UNKNOWN:
    default:
      return ErrorSeverity.MEDIUM
  }
}

// Main error processing function
export function processError(error: Error | unknown, context?: Record<string, any>): AppError {
  const errorType = categorizeError(error)
  const severity = getErrorSeverity(errorType)
  const userMessage = getUserMessage(errorType, error instanceof Error ? error.message : undefined)
  const recoveryAction = getRecoveryAction(errorType)
  
  const appError: AppError = {
    type: errorType,
    severity,
    message: error instanceof Error ? error.message : 'Unknown error occurred',
    userMessage,
    recoveryAction,
    originalError: error instanceof Error ? error : undefined,
    timestamp: new Date(),
    context
  }
  
  // Log error for debugging (in development)
  if (process.env.NODE_ENV === 'development') {
    console.group(`🚨 ${errorType.toUpperCase()} ERROR (${severity})`)
    console.error('User Message:', userMessage)
    console.error('Original Error:', error)
    console.error('Context:', context)
    console.error('Recovery Action:', recoveryAction)
    console.groupEnd()
  }
  
  return appError
}

// Specialized error processors for common scenarios
export class ErrorHandler {
  // Process API errors
  static async processApiError(response: Response, context?: Record<string, any>): Promise<AppError> {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`
    
    try {
      const errorData = await response.json()
      if (errorData.detail) {
        errorMessage = errorData.detail
      } else if (errorData.message) {
        errorMessage = errorData.message
      }
    } catch {
      // If we can't parse the error response, use the status text
    }
    
    const error = new Error(errorMessage)
    return processError(error, { ...context, status: response.status, url: response.url })
  }
  
  // Process Clerk authentication errors
  static processAuthError(error: Error, context?: Record<string, any>): AppError {
    const authContext = { ...context, component: 'clerk-auth' }
    return processError(error, authContext)
  }
  
  // Process file upload errors
  static processUploadError(error: Error, fileName?: string, context?: Record<string, any>): AppError {
    const uploadContext = { ...context, fileName, operation: 'file-upload' }
    return processError(error, uploadContext)
  }
  
  // Process chat/RAG errors
  static processChatError(error: Error, query?: string, context?: Record<string, any>): AppError {
    const chatContext = { ...context, query, operation: 'chat-query' }
    return processError(error, chatContext)
  }
}

// Hook for error handling in React components
export function useErrorHandler() {
  const handleError = (error: Error | unknown, context?: Record<string, any>) => {
    return processError(error, context)
  }
  
  const handleApiError = async (response: Response, context?: Record<string, any>) => {
    return ErrorHandler.processApiError(response, context)
  }
  
  return {
    handleError,
    handleApiError,
    processAuthError: ErrorHandler.processAuthError,
    processUploadError: ErrorHandler.processUploadError,
    processChatError: ErrorHandler.processChatError
  }
}
