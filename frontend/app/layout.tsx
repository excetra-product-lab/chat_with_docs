import React from 'react'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { ClientLayout } from '../src/components/Layout/ClientLayout'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Excetera - Chat with Your Legal Documents',
  description: 'AI-powered legal document analysis with precise sources. Upload your legal documents and get instant, accurate answers.',
  keywords: 'legal, AI, document analysis, citations, law firm, legal research',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ClientLayout>
          {children}
        </ClientLayout>
      </body>
    </html>
  )
}
