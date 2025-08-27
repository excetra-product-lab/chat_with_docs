'use client'

import React, { ReactNode } from 'react'
import { ErrorBoundary } from './ErrorBoundary'

interface DocumentErrorBoundaryProps {
  children: ReactNode
}

export function DocumentErrorBoundary({ children }: DocumentErrorBoundaryProps) {
  const handleDocumentError = (error: Error) => {
    // Log specific document-related context
    console.error('Document section error:', {
      error: error.message,
      section: 'document-management',
      timestamp: new Date().toISOString()
    })
  }

  const DocumentFallback = (
    <div className="min-h-[400px] flex items-center justify-center p-6">
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 max-w-md w-full">
        <div className="flex items-center space-x-3 mb-4">
          <div className="flex-shrink-0">
            <svg className="h-6 w-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-medium text-yellow-800">
              Document Management Unavailable
            </h3>
          </div>
        </div>
        
        <div className="mb-4">
          <p className="text-sm text-yellow-700">
            There was an issue loading the document management section. 
            You can try refreshing the page or contact support if the problem persists.
          </p>
        </div>

        <div className="flex space-x-3">
          <button
            onClick={() => window.location.reload()}
            className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-yellow-600 hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
          >
            Refresh Documents
          </button>
          <a
            href="/upload"
            className="inline-flex items-center px-3 py-2 border border-yellow-300 text-sm leading-4 font-medium rounded-md text-yellow-700 bg-white hover:bg-yellow-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
          >
            Go to Upload
          </a>
        </div>
      </div>
    </div>
  )

  return (
    <ErrorBoundary
      context="document-management"
      fallback={DocumentFallback}
      onError={handleDocumentError}
    >
      {children}
    </ErrorBoundary>
  )
}
