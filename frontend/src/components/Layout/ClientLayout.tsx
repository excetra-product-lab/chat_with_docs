'use client'

import React from 'react'
import { ClerkProvider } from '@clerk/nextjs'
import { AppProvider } from '../../context/AppContext'
import { ErrorProvider } from '../../context/ErrorContext'
import { ToastProvider } from '../Toast'
import { Navigation } from './Navigation'

export function ClientLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <ErrorProvider>
        <ToastProvider position="top-right">
          <AppProvider>
            <div className="min-h-screen bg-gradient-to-br from-midnight-950 to-slate-900">
              <Navigation />
              <main className="max-w-7xl mx-auto">
                {children}
              </main>
            </div>
          </AppProvider>
        </ToastProvider>
      </ErrorProvider>
    </ClerkProvider>
  )
}
