'use client'

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

export function Navigation() {
  const pathname = usePathname()

  return (
    <header className="px-6 py-4 border-b border-slate-800/50 bg-midnight-950/80 backdrop-blur-xl sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <Link href="/" className="flex items-center space-x-3 group">
          <div className="w-11 h-11 bg-gradient-to-br from-violet-600 via-electric-500 to-violet-700 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-glow transition-all duration-300 transform group-hover:scale-105">
            <span className="text-xl font-bold text-white">E</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Excetera</h1>
        </Link>
        <nav className="flex space-x-2">
          <Link
            href="/upload"
            className={`px-5 py-2.5 font-medium rounded-xl transition-all duration-300 ${
              pathname === '/upload'
                ? 'text-violet-300 bg-gradient-to-r from-violet-600/20 to-electric-600/20 border border-violet-500/30 shadow-glow-sm'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/50 border border-transparent hover:border-slate-700/50'
            }`}
          >
            Upload
          </Link>
          <Link
            href="/chat"
            className={`px-5 py-2.5 font-medium rounded-xl transition-all duration-300 ${
              pathname === '/chat'
                ? 'text-violet-300 bg-gradient-to-r from-violet-600/20 to-electric-600/20 border border-violet-500/30 shadow-glow-sm'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/50 border border-transparent hover:border-slate-700/50'
            }`}
          >
            Chat
          </Link>
        </nav>
      </div>
    </header>
  )
} 