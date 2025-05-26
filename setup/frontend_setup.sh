#!/bin/bash
# Frontend initialization script

# Create Next.js app (you would normally use: npx create-next-app@latest frontend --typescript --tailwind --app)
# For this script, we'll create the structure manually

# Create frontend directory structure
mkdir -p frontend/app
mkdir -p frontend/app/chat
mkdir -p frontend/app/upload
mkdir -p frontend/components
mkdir -p frontend/context
mkdir -p frontend/lib
mkdir -p frontend/styles
mkdir -p frontend/public

# Create package.json
cat > frontend/package.json << 'EOF'
{
  "name": "chat-with-docs-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@clerk/nextjs": "^4.29.3",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-label": "^2.0.2",
    "@radix-ui/react-popover": "^1.0.7",
    "@radix-ui/react-slot": "^1.0.2",
    "@radix-ui/react-toast": "^1.1.5",
    "axios": "^1.6.5",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.0",
    "lucide-react": "^0.312.0",
    "next": "14.0.4",
    "react": "^18",
    "react-dom": "^18",
    "react-dropzone": "^14.2.3",
    "tailwind-merge": "^2.2.0",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.0.1",
    "eslint": "^8",
    "eslint-config-next": "14.0.4",
    "postcss": "^8",
    "tailwindcss": "^3.3.0",
    "typescript": "^5"
  }
}
EOF

# Create tsconfig.json
cat > frontend/tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
EOF

# Create next.config.js
cat > frontend/next.config.js << 'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
}

module.exports = nextConfig
EOF

# Create app/layout.tsx
cat > frontend/app/layout.tsx << 'EOF'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { ClerkProvider } from '@clerk/nextjs'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Chat With Docs',
  description: 'AI-powered document Q&A system',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body className={inter.className}>{children}</body>
      </html>
    </ClerkProvider>
  )
}
EOF

# Create app/globals.css
cat > frontend/app/globals.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    --popover: 0 0% 100%;
    --popover-foreground: 222.2 84% 4.9%;
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;
    --radius: 0.5rem;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}
EOF

# Create app/page.tsx (landing page)
cat > frontend/app/page.tsx << 'EOF'
import { SignInButton, SignedIn, SignedOut, UserButton } from '@clerk/nextjs'
import Link from 'next/link'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold mb-8">Chat With Docs</h1>
      </div>

      <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-1 lg:text-left">
        <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100">
          <h2 className="mb-3 text-2xl font-semibold">
            AI-Powered Document Q&A
          </h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Upload your legal documents and ask questions. Get accurate answers with citations.
          </p>

          <div className="mt-6">
            <SignedOut>
              <SignInButton mode="modal">
                <button className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600">
                  Get Started
                </button>
              </SignInButton>
            </SignedOut>
            <SignedIn>
              <div className="flex gap-4">
                <Link href="/upload" className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600">
                  Upload Documents
                </Link>
                <Link href="/chat" className="bg-green-500 text-white px-6 py-2 rounded-md hover:bg-green-600">
                  Start Chatting
                </Link>
                <UserButton afterSignOutUrl="/" />
              </div>
            </SignedIn>
          </div>
        </div>
      </div>
    </main>
  )
}
EOF

# Create app/chat/page.tsx
cat > frontend/app/chat/page.tsx << 'EOF'
'use client'

import { useAuth } from '@clerk/nextjs'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import Layout from '@/components/Layout'
import ChatWindow from '@/components/ChatWindow'

export default function ChatPage() {
  const { isLoaded, isSignedIn } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (isLoaded && !isSignedIn) {
      router.push('/')
    }
  }, [isLoaded, isSignedIn, router])

  if (!isLoaded || !isSignedIn) {
    return <div>Loading...</div>
  }

  return (
    <Layout>
      <div className="container mx-auto p-4">
        <h1 className="text-2xl font-bold mb-4">Chat with your documents</h1>
        <ChatWindow />
      </div>
    </Layout>
  )
}
EOF

# Create app/upload/page.tsx
cat > frontend/app/upload/page.tsx << 'EOF'
'use client'

import { useAuth } from '@clerk/nextjs'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import Layout from '@/components/Layout'
import UploadForm from '@/components/UploadForm'
import DocumentList from '@/components/DocumentList'

export default function UploadPage() {
  const { isLoaded, isSignedIn } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (isLoaded && !isSignedIn) {
      router.push('/')
    }
  }, [isLoaded, isSignedIn, router])

  if (!isLoaded || !isSignedIn) {
    return <div>Loading...</div>
  }

  return (
    <Layout>
      <div className="container mx-auto p-4">
        <h1 className="text-2xl font-bold mb-4">Document Management</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h2 className="text-xl font-semibold mb-3">Upload New Document</h2>
            <UploadForm />
          </div>
          <div>
            <h2 className="text-xl font-semibold mb-3">Your Documents</h2>
            <DocumentList />
          </div>
        </div>
      </div>
    </Layout>
  )
}
EOF

# Create components/Layout.tsx
cat > frontend/components/Layout.tsx << 'EOF'
import { UserButton } from '@clerk/nextjs'
import Link from 'next/link'

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white shadow-sm border-b">
        <div className="container mx-auto px-4 py-3 flex justify-between items-center">
          <Link href="/" className="text-xl font-bold">
            Chat With Docs
          </Link>
          <nav className="flex items-center gap-6">
            <Link href="/upload" className="hover:text-blue-600">
              Documents
            </Link>
            <Link href="/chat" className="hover:text-blue-600">
              Chat
            </Link>
            <UserButton afterSignOutUrl="/" />
          </nav>
        </div>
      </header>
      <main className="flex-1 bg-gray-50">
        {children}
      </main>
    </div>
  )
}
EOF

# Create components/DocumentList.tsx
cat > frontend/components/DocumentList.tsx << 'EOF'
'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@clerk/nextjs'
import { api } from '@/lib/api'
import { Trash2, FileText } from 'lucide-react'

interface Document {
  id: number
  filename: string
  status: string
  created_at: string
}

export default function DocumentList() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const { getToken } = useAuth()

  useEffect(() => {
    fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    try {
      const token = await getToken()
      const response = await api.getDocuments(token!)
      setDocuments(response.data)
    } catch (error) {
      console.error('Error fetching documents:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this document?')) return

    try {
      const token = await getToken()
      await api.deleteDocument(id, token!)
      await fetchDocuments()
    } catch (error) {
      console.error('Error deleting document:', error)
    }
  }

  if (loading) return <div>Loading documents...</div>

  if (documents.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No documents uploaded yet
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className="flex items-center justify-between p-3 bg-white rounded-lg shadow-sm border"
        >
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-gray-600" />
            <div>
              <p className="font-medium">{doc.filename}</p>
              <p className="text-sm text-gray-500">
                Status: {doc.status} • {new Date(doc.created_at).toLocaleDateString()}
              </p>
            </div>
          </div>
          <button
            onClick={() => handleDelete(doc.id)}
            className="p-2 text-red-600 hover:bg-red-50 rounded"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      ))}
    </div>
  )
}
EOF

# Create components/UploadForm.tsx
cat > frontend/components/UploadForm.tsx << 'EOF'
'use client'

import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useAuth } from '@clerk/nextjs'
import { api } from '@/lib/api'
import { Upload, File } from 'lucide-react'

export default function UploadForm() {
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')
  const { getToken } = useAuth()

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return

    setUploading(true)
    setMessage('')

    try {
      const token = await getToken()
      const file = acceptedFiles[0]

      await api.uploadDocument(file, token!)
      setMessage('Document uploaded successfully!')

      // Refresh document list (you might want to use a global state or callback here)
      window.location.reload()
    } catch (error) {
      setMessage('Error uploading document')
      console.error('Upload error:', error)
    } finally {
      setUploading(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    maxFiles: 1
  })

  return (
    <div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
      >
        <input {...getInputProps()} />
        <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        {isDragActive ? (
          <p className="text-blue-600">Drop the file here...</p>
        ) : (
          <div>
            <p className="text-gray-600 mb-2">
              Drag and drop a document here, or click to select
            </p>
            <p className="text-sm text-gray-500">
              Supported formats: PDF, DOCX, TXT
            </p>
          </div>
        )}
      </div>

      {uploading && (
        <p className="mt-4 text-blue-600">Uploading...</p>
      )}

      {message && (
        <p className={`mt-4 ${message.includes('Error') ? 'text-red-600' : 'text-green-600'}`}>
          {message}
        </p>
      )}
    </div>
  )
}
EOF

# Create components/ChatWindow.tsx
cat > frontend/components/ChatWindow.tsx << 'EOF'
'use client'

import { useState, useRef, useEffect } from 'react'
import { useAuth } from '@clerk/nextjs'
import { api } from '@/lib/api'
import MessageBubble from '@/components/MessageBubble'
import { Send } from 'lucide-react'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}

interface Citation {
  document_id: number
  document_name: string
  page?: number
  snippet: string
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { getToken } = useAuth()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const token = await getToken()
      const response = await api.sendQuery(input, token!)

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.data.answer,
        citations: response.data.citations
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-[600px] bg-white rounded-lg shadow-sm border">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            Ask a question about your documents
          </div>
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg p-3">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="border-t p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </form>
    </div>
  )
}
EOF

# Create components/MessageBubble.tsx
cat > frontend/components/MessageBubble.tsx << 'EOF'
'use client'

import CitationViewer from '@/components/CitationViewer'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}

interface Citation {
  document_id: number
  document_name: string
  page?: number
  snippet: string
}

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.type === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[70%] rounded-lg p-3 ${
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-gray-100 text-gray-800'
        }`}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>

        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-sm font-semibold">Sources:</p>
            {message.citations.map((citation, index) => (
              <CitationViewer key={index} citation={citation} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
EOF

# Create components/CitationViewer.tsx
cat > frontend/components/CitationViewer.tsx << 'EOF'
'use client'

import { useState } from 'react'
import { FileText, ChevronDown, ChevronUp } from 'lucide-react'

interface Citation {
  document_id: number
  document_name: string
  page?: number
  snippet: string
}

export default function CitationViewer({ citation }: { citation: Citation }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-white/10 rounded p-2 text-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full text-left hover:bg-white/10 rounded p-1"
      >
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4" />
          <span className="font-medium">{citation.document_name}</span>
          {citation.page && <span className="text-xs opacity-75">p. {citation.page}</span>}
        </div>
        {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {expanded && (
        <div className="mt-2 p-2 bg-black/10 rounded text-xs">
          {citation.snippet}
        </div>
      )}
    </div>
  )
}
EOF

# Create lib/api.ts
cat > frontend/lib/api.ts << 'EOF'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_URL,
})

export const api = {
  // Auth endpoints
  signup: async (email: string, password: string) => {
    return apiClient.post('/api/auth/signup', { email, password })
  },

  // Document endpoints
  uploadDocument: async (file: File, token: string) => {
    const formData = new FormData()
    formData.append('file', file)

    return apiClient.post('/api/documents/upload', formData, {
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'multipart/form-data',
      },
    })
  },

  getDocuments: async (token: string) => {
    return apiClient.get('/api/documents/', {
      headers: { Authorization: `Bearer ${token}` },
    })
  },

  deleteDocument: async (id: number, token: string) => {
    return apiClient.delete(`/api/documents/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
  },

  // Chat endpoints
  sendQuery: async (question: string, token: string) => {
    return apiClient.post(
      '/api/chat/query',
      { question },
      {
        headers: { Authorization: `Bearer ${token}` },
      }
    )
  },
}
EOF

# Create middleware.ts for Clerk
cat > frontend/middleware.ts << 'EOF'
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'

const isProtectedRoute = createRouteMatcher(['/chat(.*)', '/upload(.*)'])

export default clerkMiddleware((auth, req) => {
  if (isProtectedRoute(req)) auth().protect()
})

export const config = {
  matcher: ['/((?!.*\\..*|_next).*)', '/', '/(api|trpc)(.*)'],
}
EOF

# Create tailwind.config.ts
cat > frontend/tailwind.config.ts << 'EOF'
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      animation: {
        'bounce': 'bounce 1s infinite',
      },
    },
  },
  plugins: [],
}
export default config
EOF

# Create postcss.config.js
cat > frontend/postcss.config.js << 'EOF'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOF

# Create .env.local.example
cat > frontend/.env.local.example << 'EOF'
# Clerk Authentication
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your-clerk-publishable-key
CLERK_SECRET_KEY=your-clerk-secret-key

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Clerk URLs
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/upload
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/upload
EOF

# Create frontend README
cat > frontend/README.md << 'EOF'
# Chat With Docs Frontend

## Setup

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Set up environment variables:
```bash
cp .env.local.example .env.local
# Edit .env.local with your Clerk keys and API URL
```

3. Set up Clerk:
- Go to https://clerk.dev and create a new application
- Copy your publishable key and secret key to .env.local
- Configure allowed redirect URLs in Clerk dashboard

4. Run the development server:
```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Building for Production

```bash
npm run build
npm start
```

## Deployment

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new).

Check out the [Next.js deployment documentation](https://nextjs.org/docs/deployment) for more details.
EOF

echo "Frontend structure created successfully!"
