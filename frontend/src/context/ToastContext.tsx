'use client'

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { ToastData, ToastType } from '../components/Toast/Toast'
import { ToastContainer } from '../components/Toast/ToastContainer'

interface ToastContextType {
  toasts: ToastData[]
  addToast: (toast: Omit<ToastData, 'id'>) => string
  removeToast: (id: string) => void
  clearAllToasts: () => void
  // Convenience methods
  success: (title: string, message?: string, options?: Partial<ToastData>) => string
  error: (title: string, message?: string, options?: Partial<ToastData>) => string
  warning: (title: string, message?: string, options?: Partial<ToastData>) => string
  info: (title: string, message?: string, options?: Partial<ToastData>) => string
}

const ToastContext = createContext<ToastContextType | undefined>(undefined)

interface ToastProviderProps {
  children: ReactNode
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center'
  maxToasts?: number
}

export function ToastProvider({ 
  children, 
  position = 'top-right',
  maxToasts = 5 
}: ToastProviderProps) {
  const [toasts, setToasts] = useState<ToastData[]>([])

  const addToast = useCallback((toast: Omit<ToastData, 'id'>): string => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2)}`
    
    const newToast: ToastData = {
      id,
      duration: 5000,
      persist: false,
      ...toast
    }

    setToasts(prev => {
      const updated = [newToast, ...prev]
      // Enforce max toasts limit
      return updated.slice(0, maxToasts)
    })

    return id
  }, [maxToasts])

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id))
  }, [])

  const clearAllToasts = useCallback(() => {
    setToasts([])
  }, [])

  // Convenience methods
  const success = useCallback((title: string, message?: string, options?: Partial<ToastData>) => {
    return addToast({ type: 'success', title, message, ...options })
  }, [addToast])

  const error = useCallback((title: string, message?: string, options?: Partial<ToastData>) => {
    return addToast({ 
      type: 'error', 
      title, 
      message, 
      duration: 7000, // Errors stay longer by default
      ...options 
    })
  }, [addToast])

  const warning = useCallback((title: string, message?: string, options?: Partial<ToastData>) => {
    return addToast({ 
      type: 'warning', 
      title, 
      message, 
      duration: 6000, // Warnings stay slightly longer
      ...options 
    })
  }, [addToast])

  const info = useCallback((title: string, message?: string, options?: Partial<ToastData>) => {
    return addToast({ type: 'info', title, message, ...options })
  }, [addToast])

  const contextValue: ToastContextType = {
    toasts,
    addToast,
    removeToast,
    clearAllToasts,
    success,
    error,
    warning,
    info
  }

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      <ToastContainer 
        toasts={toasts}
        onRemoveToast={removeToast}
        position={position}
      />
    </ToastContext.Provider>
  )
}

export function useToast(): ToastContextType {
  const context = useContext(ToastContext)
  if (context === undefined) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}

// Hook for showing toasts with error context integration
export function useToastWithErrorHandling() {
  const toast = useToast()

  const showApiError = useCallback((error: any, operation: string) => {
    if (error?.userMessage) {
      toast.error(error.userMessage, error.recoveryAction)
    } else {
      toast.error(
        `${operation} failed`,
        'Please try again or contact support if the problem persists.'
      )
    }
  }, [toast])

  const showSuccess = useCallback((title: string, message?: string) => {
    toast.success(title, message)
  }, [toast])

  return {
    ...toast,
    showApiError,
    showSuccess
  }
}

