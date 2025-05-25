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
