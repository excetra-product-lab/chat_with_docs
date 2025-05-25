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
