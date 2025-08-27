'use client'

import React, { useEffect, useState } from 'react'
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface ToastData {
  id: string
  type: ToastType
  title: string
  message?: string
  duration?: number
  persist?: boolean
}

interface ToastProps {
  toast: ToastData
  onRemove: (id: string) => void
}

export function Toast({ toast, onRemove }: ToastProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isLeaving, setIsLeaving] = useState(false)

  useEffect(() => {
    // Trigger entrance animation
    const timer1 = setTimeout(() => setIsVisible(true), 50)
    
    // Auto-dismiss if not persistent
    let timer2: NodeJS.Timeout
    if (!toast.persist) {
      timer2 = setTimeout(() => {
        handleRemove()
      }, toast.duration || 5000)
    }

    return () => {
      clearTimeout(timer1)
      if (timer2) clearTimeout(timer2)
    }
  }, [toast.duration, toast.persist])

  const handleRemove = () => {
    setIsLeaving(true)
    setTimeout(() => onRemove(toast.id), 300) // Allow animation to complete
  }

  const typeConfig = {
    success: {
      icon: CheckCircle,
      containerClass: 'bg-green-500/10 border-green-500/30 shadow-green-500/20',
      iconClass: 'text-green-400',
      titleClass: 'text-green-200',
      messageClass: 'text-green-300'
    },
    error: {
      icon: AlertCircle,
      containerClass: 'bg-red-500/10 border-red-500/30 shadow-red-500/20',
      iconClass: 'text-red-400', 
      titleClass: 'text-red-200',
      messageClass: 'text-red-300'
    },
    warning: {
      icon: AlertTriangle,
      containerClass: 'bg-yellow-500/10 border-yellow-500/30 shadow-yellow-500/20',
      iconClass: 'text-yellow-400',
      titleClass: 'text-yellow-200', 
      messageClass: 'text-yellow-300'
    },
    info: {
      icon: Info,
      containerClass: 'bg-blue-500/10 border-blue-500/30 shadow-blue-500/20',
      iconClass: 'text-blue-400',
      titleClass: 'text-blue-200',
      messageClass: 'text-blue-300'
    }
  }

  const config = typeConfig[toast.type]
  const Icon = config.icon

  return (
    <div
      className={`
        relative flex items-start space-x-3 p-4 rounded-lg border backdrop-blur-sm
        transition-all duration-300 ease-out transform
        ${config.containerClass}
        ${isVisible && !isLeaving 
          ? 'translate-x-0 opacity-100 scale-100' 
          : 'translate-x-full opacity-0 scale-95'
        }
      `}
    >
      <div className="flex-shrink-0">
        <Icon className={`w-5 h-5 ${config.iconClass}`} />
      </div>
      
      <div className="flex-1 min-w-0">
        <p className={`text-sm font-medium ${config.titleClass}`}>
          {toast.title}
        </p>
        {toast.message && (
          <p className={`text-sm mt-1 ${config.messageClass}`}>
            {toast.message}
          </p>
        )}
      </div>

      <button
        onClick={handleRemove}
        className="flex-shrink-0 p-1 rounded-full hover:bg-white/10 transition-colors"
      >
        <X className="w-4 h-4 text-slate-400 hover:text-slate-200" />
      </button>
    </div>
  )
}

