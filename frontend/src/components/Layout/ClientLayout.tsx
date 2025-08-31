'use client'

import React, { useEffect } from 'react'
import { ClerkProvider } from '@clerk/nextjs'
import { AppProvider } from '../../context/AppContext'
import { ErrorProvider } from '../../context/ErrorContext'
import { ToastProvider } from '../Toast'
import { Navigation } from './Navigation'
import { validateEnvironment, isDevelopment } from '../../lib/env-validation'

export function ClientLayout({ children }: { children: React.ReactNode }) {
  // Validate environment variables in development
  useEffect(() => {
    if (isDevelopment()) {
      const validation = validateEnvironment()
      
      if (!validation.isValid) {
        console.error('❌ Environment validation failed!')
        validation.errors.forEach((error: string) => console.error(`  • ${error}`))
        console.error('\nPlease check your .env.local file and ensure all required variables are set.')
        console.error('See the README.md for setup instructions.')
      } else if (validation.warnings.length > 0) {
        console.warn('⚠️ Environment warnings:')
        validation.warnings.forEach((warning: string) => console.warn(`  • ${warning}`))
      } else {
        console.log('✅ Environment validation passed')
      }
    }
  }, [])

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
