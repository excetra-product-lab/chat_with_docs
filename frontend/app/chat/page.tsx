'use client'

import React from 'react'
import { ChatInterface } from '../../src/components/ChatInterface'
import { useAppContext } from '../../src/context/AppContext'

export default function ChatPage() {
  const { messages, isLoading, addMessage, documents } = useAppContext()

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

    // Mock AI response - in real app this would call the backend API
    setTimeout(() => {
      const aiMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant' as const,
        content: `I understand you're asking about: "${content}". I would analyze your uploaded documents and provide relevant information here.`,
        timestamp: new Date()
        // citations field removed
      }
      addMessage(aiMessage)
    }, 1000)
  }

  return (
    <div className="h-screen px-6">
      <ChatInterface
        messages={messages}
        isLoading={isLoading}
        onSendMessage={sendMessage}
        hasDocuments={hasReadyDocuments}
      />
    </div>
  )
}
