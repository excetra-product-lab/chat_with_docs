'use client'

import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react'

// Import types from existing type definitions
import { Document, Message } from '../types'
import { useDocuments } from '../hooks/useDocuments'

// Extend the existing types for our context
export interface DocumentWithProgress extends Document {
  uploadProgress?: number
}

interface AppContextType {
  // Document state
  documents: DocumentWithProgress[]
  setDocuments: (documents: DocumentWithProgress[]) => void
  addDocument: (document: DocumentWithProgress) => void
  updateDocument: (id: number, updates: Partial<DocumentWithProgress>) => void
  deleteDocument: (id: number) => void

  // Chat state
  messages: Message[]
  setMessages: (messages: Message[]) => void
  addMessage: (message: Message) => void

  // UI state
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
  
  // Document loading state
  isDocumentsLoading: boolean
  documentsError: string | null
  refetchDocuments: () => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  // Use the existing useDocuments hook to get real data
  const { documents: fetchedDocuments, isLoading: isDocumentsLoading, error: documentsError, refetch: refetchDocuments } = useDocuments()
  
  const [documents, setDocuments] = useState<DocumentWithProgress[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  
  // Sync fetched documents with local state
  useEffect(() => {
    setDocuments(fetchedDocuments)
  }, [fetchedDocuments])

  const addDocument = (document: DocumentWithProgress) => {
    setDocuments(prev => [...prev, document])
  }

  const updateDocument = (id: number, updates: Partial<DocumentWithProgress>) => {
    setDocuments(prev =>
      prev.map(doc => doc.id === id ? { ...doc, ...updates } : doc)
    )
  }

  const deleteDocument = (id: number) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id))
    // Citation filtering removed - messages are no longer linked to documents via citations
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
    
    // Document loading state
    isDocumentsLoading,
    documentsError,
    refetchDocuments,
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
