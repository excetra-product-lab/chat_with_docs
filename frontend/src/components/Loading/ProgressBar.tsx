'use client'

import React from 'react'

interface ProgressBarProps {
  progress: number // 0-100
  className?: string
  showPercentage?: boolean
  color?: 'primary' | 'success' | 'warning' | 'error'
  size?: 'small' | 'medium' | 'large'
}

export function ProgressBar({
  progress,
  className = '',
  showPercentage = false,
  color = 'primary',
  size = 'medium'
}: ProgressBarProps) {
  const colorClasses = {
    primary: 'bg-blue-500',
    success: 'bg-green-500', 
    warning: 'bg-yellow-500',
    error: 'bg-red-500'
  }

  const sizeClasses = {
    small: 'h-1',
    medium: 'h-2',
    large: 'h-3'
  }

  const backgroundClasses = {
    primary: 'bg-blue-500/20',
    success: 'bg-green-500/20',
    warning: 'bg-yellow-500/20', 
    error: 'bg-red-500/20'
  }

  const clampedProgress = Math.min(100, Math.max(0, progress))

  return (
    <div className={`w-full ${className}`}>
      {showPercentage && (
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm text-slate-400">Progress</span>
          <span className="text-sm text-slate-300">{Math.round(clampedProgress)}%</span>
        </div>
      )}
      <div className={`w-full ${backgroundClasses[color]} rounded-full ${sizeClasses[size]}`}>
        <div
          className={`${colorClasses[color]} ${sizeClasses[size]} rounded-full transition-all duration-300 ease-out`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
    </div>
  )
}

export function CircularProgress({
  progress,
  size = 40,
  strokeWidth = 4,
  color = 'primary',
  showPercentage = false,
  className = ''
}: {
  progress: number
  size?: number
  strokeWidth?: number
  color?: 'primary' | 'success' | 'warning' | 'error'
  showPercentage?: boolean
  className?: string
}) {
  const colorClasses = {
    primary: 'stroke-blue-500',
    success: 'stroke-green-500',
    warning: 'stroke-yellow-500', 
    error: 'stroke-red-500'
  }

  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const clampedProgress = Math.min(100, Math.max(0, progress))
  const offset = circumference - (clampedProgress / 100) * circumference

  return (
    <div className={`relative inline-flex items-center justify-center ${className}`}>
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          className="text-slate-700"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className={`${colorClasses[color]} transition-all duration-300 ease-out`}
          strokeLinecap="round"
        />
      </svg>
      {showPercentage && (
        <span className="absolute text-xs font-medium text-slate-300">
          {Math.round(clampedProgress)}%
        </span>
      )}
    </div>
  )
}

