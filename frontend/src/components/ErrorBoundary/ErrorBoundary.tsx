'use client'

import React, { ErrorInfo, ReactNode } from 'react'
import { useError } from '../../context/ErrorContext'
import { processError, ErrorType } from '../../lib/errorHandler'

interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
}

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
  context?: string
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo
    })

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    }

    // Process the error for logging
    processError(error, {
      component: 'ErrorBoundary',
      context: this.props.context,
      componentStack: errorInfo.componentStack
    })
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined })
  }

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback
      }

      // Default fallback UI
      return (
        <ErrorFallback
          error={this.state.error}
          context={this.props.context}
          onRetry={this.handleRetry}
        />
      )
    }

    return this.props.children
  }
}

interface ErrorFallbackProps {
  error?: Error
  context?: string
  onRetry?: () => void
}

function ErrorFallback({ error, context, onRetry }: ErrorFallbackProps) {
  return (
    <div className="min-h-[200px] flex items-center justify-center p-6">
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md w-full">
        <div className="flex items-center space-x-3 mb-4">
          <div className="flex-shrink-0">
            <svg className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 19.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-medium text-red-800">
              Something went wrong{context && ` in ${context}`}
            </h3>
          </div>
        </div>
        
        <div className="mb-4">
          <p className="text-sm text-red-700">
            An unexpected error occurred while rendering this component. 
            This might be due to a temporary issue.
          </p>
          {process.env.NODE_ENV === 'development' && error && (
            <details className="mt-2">
              <summary className="text-xs text-red-600 cursor-pointer">Error Details (Development)</summary>
              <pre className="text-xs text-red-600 mt-1 whitespace-pre-wrap break-all">
                {error.message}
                {error.stack && `\n\n${error.stack}`}
              </pre>
            </details>
          )}
        </div>

        <div className="flex space-x-3">
          {onRetry && (
            <button
              onClick={onRetry}
              className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
            >
              Try Again
            </button>
          )}
          <button
            onClick={() => window.location.reload()}
            className="inline-flex items-center px-3 py-2 border border-red-300 text-sm leading-4 font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
          >
            Reload Page
          </button>
        </div>
      </div>
    </div>
  )
}

// Hook version for functional components
export function useErrorBoundary() {
  const { addError } = useError()
  
  const resetError = React.useCallback(() => {
    // This will be used by error boundaries to reset their state
  }, [])

  const captureError = React.useCallback((error: Error, context?: Record<string, any>) => {
    addError(error, { ...context, capturedBy: 'useErrorBoundary' })
  }, [addError])

  return { resetError, captureError }
}
