'use client'

import React from 'react'

interface SkeletonLoaderProps {
  className?: string
  width?: string | number
  height?: string | number
  rounded?: boolean
}

export function SkeletonLoader({ 
  className = '', 
  width = '100%', 
  height = '1rem',
  rounded = false 
}: SkeletonLoaderProps) {
  const style = {
    width: typeof width === 'number' ? `${width}px` : width,
    height: typeof height === 'number' ? `${height}px` : height,
  }

  return (
    <div
      className={`animate-pulse bg-slate-300/20 ${rounded ? 'rounded-full' : 'rounded'} ${className}`}
      style={style}
    />
  )
}

// Specific skeleton components for common patterns
export function DocumentSkeleton() {
  return (
    <div className="space-y-3 p-4 border border-slate-700/30 rounded-lg">
      <div className="flex items-center space-x-3">
        <SkeletonLoader width={24} height={24} rounded />
        <SkeletonLoader width="60%" height="1.25rem" />
      </div>
      <div className="space-y-2">
        <SkeletonLoader width="40%" height="0.875rem" />
        <SkeletonLoader width="30%" height="0.875rem" />
      </div>
    </div>
  )
}

export function MessageSkeleton() {
  return (
    <div className="space-y-2">
      <div className="flex items-center space-x-2">
        <SkeletonLoader width={32} height={32} rounded />
        <SkeletonLoader width="20%" height="1rem" />
      </div>
      <div className="ml-10 space-y-2">
        <SkeletonLoader width="80%" height="1rem" />
        <SkeletonLoader width="60%" height="1rem" />
        <SkeletonLoader width="40%" height="1rem" />
      </div>
    </div>
  )
}

export function TableRowSkeleton({ columns = 4 }: { columns?: number }) {
  return (
    <tr className="border-b border-slate-700/30">
      {Array.from({ length: columns }).map((_, index) => (
        <td key={index} className="px-4 py-3">
          <SkeletonLoader height="1rem" width={index === 0 ? "80%" : "60%"} />
        </td>
      ))}
    </tr>
  )
}

