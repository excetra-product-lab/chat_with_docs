'use client'

import React, { useState } from 'react'
import { ChatInterface } from '../../src/components/ChatInterface'
import { useAppContext } from '../../src/context/AppContext'
import { useApi } from '../../src/lib/api'
import { ChatErrorBoundary } from '../../src/components/ErrorBoundary'

export default function ChatPage() {
  const { messages, isLoading, addMessage, documents } = useAppContext()
  const [isSending, setIsSending] = useState(false)
  const api = useApi()

  const hasReadyDocuments = documents.some(doc => doc.status === 'ready')

  const sendMessage = async (content: string) => {
    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      type: 'user' as const,
      content,
      timestamp: new Date(),
    }
    addMessage(userMessage)

    setIsSending(true)

    try {
      // Check if user has documents before making API call
      if (!hasReadyDocuments) {
        const noDocsMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant' as const,
          content: "I don't have any documents to search through yet. Please upload some documents first, and I'll be able to help answer questions about them.",
          timestamp: new Date(),
          confidence: 0
        }
        addMessage(noDocsMessage)
        return
      }

      // Real RAG API call
      const response = await api.sendQuery(content)
      
      const aiMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant' as const,
        content: response.answer,
        timestamp: new Date(),
        confidence: response.confidence
      }
      addMessage(aiMessage)
    } catch (error) {
      console.error('Chat query failed:', error)
      
      let errorContent = "I apologize, but I encountered an error while processing your question."
      
      // Provide specific error messages based on error type
      if (error instanceof Error) {
        if (error.message.includes('401') || error.message.includes('Unauthorized')) {
          errorContent = "It looks like you're not signed in properly. Please try refreshing the page or signing in again."
        } else if (error.message.includes('404')) {
          errorContent = "The chat service is currently unavailable. Please try again in a few moments."
        } else if (error.message.includes('500')) {
          errorContent = "Our servers are experiencing issues. Please try again in a few moments."
        } else if (error.message.includes('Network')) {
          errorContent = "There seems to be a network connectivity issue. Please check your internet connection and try again."
        } else {
          errorContent = `${errorContent} ${error.message}`
        }
      }
      
      // Add helpful recovery suggestions
      errorContent += " If the problem persists, try uploading your documents again or contact support."
      
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant' as const,
        content: errorContent,
        timestamp: new Date(),
        confidence: 0
      }
      addMessage(errorMessage)
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="h-screen px-6">
      <ChatErrorBoundary>
        <ChatInterface
          messages={messages}
          isLoading={isLoading || isSending}
          onSendMessage={sendMessage}
          hasDocuments={hasReadyDocuments}
        />
      </ChatErrorBoundary>
    </div>
  )
}
