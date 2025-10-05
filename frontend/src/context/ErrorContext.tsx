'use client'

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { AppError, processError } from '../lib/errorHandler'

interface ErrorContextType {
  errors: AppError[]
  addError: (error: Error | unknown, context?: Record<string, any>) => AppError
  removeError: (timestamp: Date) => void
  clearErrors: () => void
  hasErrors: boolean
}

const ErrorContext = createContext<ErrorContextType | undefined>(undefined)

interface ErrorProviderProps {
  children: ReactNode
}

export function ErrorProvider({ children }: ErrorProviderProps) {
  const [errors, setErrors] = useState<AppError[]>([])

  const addError = useCallback((error: Error | unknown, context?: Record<string, any>): AppError => {
    const appError = processError(error, context)
    
    setErrors(prev => {
      // Prevent duplicate errors by checking recent errors
      const isDuplicate = prev.some(existingError => 
        existingError.message === appError.message &&
        Date.now() - existingError.timestamp.getTime() < 5000 // 5 seconds
      )
      
      if (isDuplicate) {
        return prev
      }
      
      // Keep only the last 10 errors to prevent memory issues
      const newErrors = [appError, ...prev].slice(0, 10)
      return newErrors
    })
    
    return appError
  }, [])

  const removeError = useCallback((timestamp: Date) => {
    setErrors(prev => prev.filter(error => error.timestamp !== timestamp))
  }, [])

  const clearErrors = useCallback(() => {
    setErrors([])
  }, [])

  const hasErrors = errors.length > 0

  const value: ErrorContextType = {
    errors,
    addError,
    removeError,
    clearErrors,
    hasErrors
  }

  return (
    <ErrorContext.Provider value={value}>
      {children}
    </ErrorContext.Provider>
  )
}

export function useError() {
  const context = useContext(ErrorContext)
  if (context === undefined) {
    throw new Error('useError must be used within an ErrorProvider')
  }
  return context
}

// Convenience hook for adding specific error types
export function useErrorActions() {
  const { addError } = useError()
  
  const addNetworkError = useCallback((error: Error | unknown) => {
    return addError(error, { type: 'network' })
  }, [addError])
  
  const addAuthError = useCallback((error: Error | unknown) => {
    return addError(error, { type: 'authentication' })
  }, [addError])
  
  const addValidationError = useCallback((error: Error | unknown) => {
    return addError(error, { type: 'validation' })
  }, [addError])
  
  const addServerError = useCallback((error: Error | unknown) => {
    return addError(error, { type: 'server' })
  }, [addError])
  
  return {
    addError,
    addNetworkError,
    addAuthError,
    addValidationError,
    addServerError
  }
}
