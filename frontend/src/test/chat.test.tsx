import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import { MessageBubble } from '../components/MessageBubble'
import { Message } from '../types'

describe('MessageBubble', () => {
  const mockUserMessage: Message = {
    id: '1',
    type: 'user',
    content: 'What is this document about?',
    timestamp: new Date('2024-01-01T12:00:00Z'),
  }

  const mockAssistantMessage: Message = {
    id: '2',
    type: 'assistant',
    content: 'This document is about legal contracts.',
    timestamp: new Date('2024-01-01T12:01:00Z'),
    confidence: 0.85,
  }

  const mockAssistantWithCitations: Message = {
    id: '3',
    type: 'assistant',
    content: 'Based on the uploaded documents, this contract includes termination clauses.',
    timestamp: new Date('2024-01-01T12:02:00Z'),
    confidence: 0.92,
    citations: [
      {
        document_name: 'contract.pdf',
        page_number: 3,
        chunk_text: 'Either party may terminate this agreement with 30 days written notice.',
        chunk_index: 5,
      },
      {
        document_name: 'legal_terms.pdf',
        page_number: 1,
        chunk_text: 'Termination procedures must follow applicable law.',
        chunk_index: 2,
      }
    ]
  }

  beforeEach(() => {
    vi.clearAllMocks()
    cleanup()
  })

  it('should render user message correctly', () => {
    render(<MessageBubble message={mockUserMessage} />)
    
    expect(screen.getByText('What is this document about?')).toBeInTheDocument()
    expect(screen.getByText('12:00 PM')).toBeInTheDocument()
  })

  it('should render assistant message with confidence score', () => {
    render(<MessageBubble message={mockAssistantMessage} />)
    
    expect(screen.getByText('This document is about legal contracts.')).toBeInTheDocument()
    expect(screen.getByText('Confidence:')).toBeInTheDocument()
    expect(screen.getByText('85%')).toBeInTheDocument()
    
    // Check for copy and regenerate buttons
    expect(screen.getByTitle('Copy message')).toBeInTheDocument()
    expect(screen.getByTitle('Regenerate response')).toBeInTheDocument()
  })

  it('should display citations when available', () => {
    render(<MessageBubble message={mockAssistantWithCitations} />)
    
    expect(screen.getByText('Sources (2)')).toBeInTheDocument()
    
    // Citations should be collapsed by default
    expect(screen.queryByText('contract.pdf')).not.toBeInTheDocument()
  })

  it('should expand citations when clicked', async () => {
    render(<MessageBubble message={mockAssistantWithCitations} />)
    
    const sourcesButton = screen.getByText('Sources (2)')
    fireEvent.click(sourcesButton)
    
    await waitFor(() => {
      expect(screen.getByText('contract.pdf')).toBeInTheDocument()
      expect(screen.getByText('legal_terms.pdf')).toBeInTheDocument()
      expect(screen.getByText('Page 3')).toBeInTheDocument()
      expect(screen.getByText('Page 1')).toBeInTheDocument()
      expect(screen.getByText('"Either party may terminate this agreement with 30 days written notice."')).toBeInTheDocument()
    })
  })

  it('should call onCopy when copy button is clicked', () => {
    const mockOnCopy = vi.fn()
    
    // Mock clipboard API
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    })
    
    render(<MessageBubble message={mockAssistantMessage} onCopy={mockOnCopy} />)
    
    const copyButton = screen.getByTitle('Copy message')
    fireEvent.click(copyButton)
    
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('This document is about legal contracts.')
    expect(mockOnCopy).toHaveBeenCalled()
  })

  it('should call onRegenerate when regenerate button is clicked', () => {
    const mockOnRegenerate = vi.fn()
    
    render(<MessageBubble message={mockAssistantMessage} onRegenerate={mockOnRegenerate} />)
    
    const regenerateButton = screen.getByTitle('Regenerate response')
    fireEvent.click(regenerateButton)
    
    expect(mockOnRegenerate).toHaveBeenCalled()
  })

  it('should display confidence score with correct color coding', () => {
    // High confidence (green)
    const highConfidenceMessage = { ...mockAssistantMessage, confidence: 0.9 }
    const { rerender } = render(<MessageBubble message={highConfidenceMessage} />)
    
    let progressBar = document.querySelector('.bg-green-500')
    expect(progressBar).toBeInTheDocument()
    
    // Medium confidence (yellow)
    const mediumConfidenceMessage = { ...mockAssistantMessage, confidence: 0.7 }
    rerender(<MessageBubble message={mediumConfidenceMessage} />)
    
    progressBar = document.querySelector('.bg-yellow-500')
    expect(progressBar).toBeInTheDocument()
    
    // Low confidence (red)
    const lowConfidenceMessage = { ...mockAssistantMessage, confidence: 0.5 }
    rerender(<MessageBubble message={lowConfidenceMessage} />)
    
    progressBar = document.querySelector('.bg-red-500')
    expect(progressBar).toBeInTheDocument()
  })

  it('should not show confidence score for user messages', () => {
    render(<MessageBubble message={mockUserMessage} />)
    
    expect(screen.queryByText('Confidence:')).not.toBeInTheDocument()
  })

  it('should not show copy/regenerate buttons for user messages', () => {
    render(<MessageBubble message={mockUserMessage} />)
    
    expect(screen.queryByTitle('Copy message')).not.toBeInTheDocument()
    expect(screen.queryByTitle('Regenerate response')).not.toBeInTheDocument()
  })
})
