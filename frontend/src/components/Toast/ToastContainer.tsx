'use client'

import React from 'react'
import { Toast, ToastData } from './Toast'

interface ToastContainerProps {
  toasts: ToastData[]
  onRemoveToast: (id: string) => void
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center'
}

export function ToastContainer({ 
  toasts, 
  onRemoveToast, 
  position = 'top-right' 
}: ToastContainerProps) {
  const positionClasses = {
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'top-center': 'top-4 left-1/2 transform -translate-x-1/2',
    'bottom-center': 'bottom-4 left-1/2 transform -translate-x-1/2'
  }

  if (toasts.length === 0) {
    return null
  }

  return (
    <div
      className={`fixed z-50 pointer-events-none ${positionClasses[position]}`}
      style={{ maxWidth: '420px', width: 'calc(100vw - 2rem)' }}
    >
      <div className="space-y-3 pointer-events-auto">
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            toast={toast}
            onRemove={onRemoveToast}
          />
        ))}
      </div>
    </div>
  )
}

