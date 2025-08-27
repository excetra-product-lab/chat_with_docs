'use client'

import React, { ReactNode } from 'react'
import { ErrorBoundary } from './ErrorBoundary'

interface ChatErrorBoundaryProps {
  children: ReactNode
}

export function ChatErrorBoundary({ children }: ChatErrorBoundaryProps) {
  const handleChatError = (error: Error) => {
    // Log specific chat-related context
    console.error('Chat section error:', {
      error: error.message,
      section: 'chat-interface',
      timestamp: new Date().toISOString()
    })
  }

  const ChatFallback = (
    <div className="min-h-[500px] flex items-center justify-center p-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 max-w-md w-full">
        <div className="flex items-center space-x-3 mb-4">
          <div className="flex-shrink-0">
            <svg className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-medium text-blue-800">
              Chat Interface Unavailable
            </h3>
          </div>
        </div>
        
        <div className="mb-4">
          <p className="text-sm text-blue-700">
            There was an issue loading the chat interface. This might be a temporary problem 
            with the AI service or your documents.
          </p>
        </div>

        <div className="flex space-x-3">
          <button
            onClick={() => window.location.reload()}
            className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Refresh Chat
          </button>
          <a
            href="/upload"
            className="inline-flex items-center px-3 py-2 border border-blue-300 text-sm leading-4 font-medium rounded-md text-blue-700 bg-white hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Check Documents
          </a>
        </div>
      </div>
    </div>
  )

  return (
    <ErrorBoundary
      context="chat-interface"
      fallback={ChatFallback}
      onError={handleChatError}
    >
      {children}
    </ErrorBoundary>
  )
}
