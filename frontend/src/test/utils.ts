import { vi } from 'vitest'
import type { Document } from '../types'

// Mock Clerk authentication
export const mockClerkAuth = {
  signedIn: {
    getToken: vi.fn(() => Promise.resolve('mock-token-123')),
    isSignedIn: true,
    userId: 'user_123',
  },
  signedOut: {
    getToken: vi.fn(() => Promise.resolve(null)),
    isSignedIn: false,
    userId: null,
  },
  expired: {
    getToken: vi.fn(() => Promise.reject(new Error('Token expired'))),
    isSignedIn: true,
    userId: 'user_123',
  },
}

// Mock documents for testing
export const mockDocuments: Document[] = [
  {
    id: 1,
    filename: 'test-document.pdf',
    status: 'processed',
    file_size: 1024000,
    created_at: '2024-01-01T12:00:00Z',
    file_type: 'application/pdf',
    user_id: 'user_123',
  },
  {
    id: 2,
    filename: 'processing-document.pdf',
    status: 'processing',
    file_size: 2048000,
    created_at: '2024-01-01T12:30:00Z',
    file_type: 'application/pdf',
    user_id: 'user_123',
  },
  {
    id: 3,
    filename: 'failed-document.pdf',
    status: 'failed',
    file_size: 512000,
    created_at: '2024-01-01T11:00:00Z',
    file_type: 'application/pdf',
    user_id: 'user_123',
  },
]

// Mock API responses
export const mockApiResponses = {
  documents: {
    success: mockDocuments,
    empty: [],
    single: mockDocuments[0],
  },
  chat: {
    success: {
      answer: 'This is a test response based on your documents.',
      confidence: 0.85,
      citations: [
        {
          document_name: 'test-document.pdf',
          page_number: 1,
          chunk_text: 'This is a sample chunk from the document.',
          chunk_index: 0,
        },
      ],
      query: 'What is this document about?',
      retrieved_chunks: [
        {
          document_name: 'test-document.pdf',
          chunk_text: 'This is a sample chunk from the document.',
          page_number: 1,
          chunk_index: 0,
        },
      ],
    },
    lowConfidence: {
      answer: 'I\'m not very confident about this answer.',
      confidence: 0.3,
      citations: [],
      query: 'Unclear question',
      retrieved_chunks: [],
    },
  },
  upload: {
    success: {
      id: 4,
      name: 'uploaded-document.pdf',
      status: 'processing',
      size: 1024000,
      uploaded_at: '2024-01-01T12:00:00Z',
      processed_at: null,
      chunk_count: 0,
      user_id: 'user_123',
    },
  },
  errors: {
    unauthorized: {
      status: 401,
      statusText: 'Unauthorized',
      body: { detail: 'Authentication required' },
    },
    notFound: {
      status: 404,
      statusText: 'Not Found',
      body: { detail: 'Document not found' },
    },
    serverError: {
      status: 500,
      statusText: 'Internal Server Error',
      body: { detail: 'Internal server error' },
    },
    validationError: {
      status: 422,
      statusText: 'Unprocessable Entity',
      body: { detail: 'Invalid file format' },
    },
  },
}

// Mock fetch responses
export const createMockResponse = (data: any, status = 200, statusText = 'OK') => {
  return new Response(JSON.stringify(data), {
    status,
    statusText,
    headers: {
      'Content-Type': 'application/json',
    },
  })
}

export const createMockErrorResponse = (status: number, statusText: string, body: any) => {
  return new Response(JSON.stringify(body), {
    status,
    statusText,
    headers: {
      'Content-Type': 'application/json',
    },
  })
}

// Mock XMLHttpRequest for upload testing
export const createMockXMLHttpRequest = (options: {
  shouldSucceed?: boolean
  shouldProgress?: boolean
  responseData?: any
  errorStatus?: number
  errorMessage?: string
}) => {
  const {
    shouldSucceed = true,
    shouldProgress = true,
    responseData = mockApiResponses.upload.success,
    errorStatus = 400,
    errorMessage = 'Upload failed',
  } = options

  const mockXHR = {
    upload: {
      addEventListener: vi.fn((event: string, callback: Function) => {
        if (event === 'progress' && shouldProgress) {
          // Simulate upload progress
          setTimeout(() => callback({ lengthComputable: true, loaded: 50, total: 100 }), 10)
          setTimeout(() => callback({ lengthComputable: true, loaded: 100, total: 100 }), 20)
        }
      }),
    },
    addEventListener: vi.fn((event: string, callback: Function) => {
      if (event === 'load') {
        setTimeout(() => {
          mockXHR.status = shouldSucceed ? 200 : errorStatus
          mockXHR.statusText = shouldSucceed ? 'OK' : errorMessage
          mockXHR.responseText = shouldSucceed 
            ? JSON.stringify(responseData)
            : JSON.stringify({ detail: errorMessage })
          callback()
        }, 30)
      } else if (event === 'error' && !shouldSucceed) {
        setTimeout(() => callback(), 30)
      }
    }),
    open: vi.fn(),
    setRequestHeader: vi.fn(),
    send: vi.fn(),
    status: 0,
    statusText: '',
    responseText: '',
  }

  return mockXHR
}

// Setup global mocks
export const setupGlobalMocks = () => {
  // Mock environment variables
  process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000'
  process.env.NEXT_PUBLIC_API_BASE_URL = 'http://localhost:8000'

  // Mock fetch globally
  global.fetch = vi.fn()

  // Mock XMLHttpRequest globally
  global.XMLHttpRequest = vi.fn(() => createMockXMLHttpRequest({})) as any

  // Mock clipboard API
  Object.assign(navigator, {
    clipboard: {
      writeText: vi.fn().mockResolvedValue(undefined),
    },
  })

  // Mock URL.createObjectURL and revokeObjectURL
  global.URL.createObjectURL = vi.fn(() => 'mock-object-url')
  global.URL.revokeObjectURL = vi.fn()
}

// Test helper to wait for async operations
export const waitForAsync = (ms = 0) => new Promise(resolve => setTimeout(resolve, ms))

// Reset all mocks
export const resetAllMocks = () => {
  vi.clearAllMocks()
  
  if (global.fetch) {
    (global.fetch as any).mockReset()
  }
}
