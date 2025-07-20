'use client'

import React, { createContext, useContext, useState, ReactNode } from 'react'

// Import types from existing type definitions
import { Document, Message } from '../types'

// Extend the existing types for our context
export interface DocumentWithProgress extends Document {
  uploadProgress?: number
}

interface AppContextType {
  // Document state
  documents: DocumentWithProgress[]
  setDocuments: (documents: DocumentWithProgress[]) => void
  addDocument: (document: DocumentWithProgress) => void
  updateDocument: (id: string, updates: Partial<DocumentWithProgress>) => void
  deleteDocument: (id: string) => void
  
  // Chat state
  messages: Message[]
  setMessages: (messages: Message[]) => void
  addMessage: (message: Message) => void
  
  // UI state
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  const [documents, setDocuments] = useState<DocumentWithProgress[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const addDocument = (document: DocumentWithProgress) => {
    setDocuments(prev => [...prev, document])
  }

  const updateDocument = (id: string, updates: Partial<DocumentWithProgress>) => {
    setDocuments(prev => 
      prev.map(doc => doc.id === id ? { ...doc, ...updates } : doc)
    )
  }

  const deleteDocument = (id: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id))
    // Also remove related chat messages if needed
    setMessages(prev => prev.filter(msg => 
      !msg.citations?.some(citation => citation.documentId === id)
    ))
  }

  const addMessage = (message: Message) => {
    setMessages(prev => [...prev, message])
  }

  const value: AppContextType = {
    // Document state
    documents,
    setDocuments,
    addDocument,
    updateDocument,
    deleteDocument,
    
    // Chat state
    messages,
    setMessages,
    addMessage,
    
    // UI state
    isLoading,
    setIsLoading,
  }

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  )
}

export function useAppContext() {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider')
  }
  return context
} 