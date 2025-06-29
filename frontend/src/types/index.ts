export interface Document {
  id: string;
  filename: string;
  user_id: string;
  status: 'processing' | 'ready' | 'failed';
  created_at: string;
  file_size: number;
  file_type: string;
  pages?: number;
  upload_progress?: number;
}

export interface DocumentChunk {
  id: string;
  documentId: string;
  content: string;
  page?: number;
  section?: string;
  startIndex: number;
  endIndex: number;
}

export interface Citation {
  documentId: string;
  documentName: string;
  chunkId?: string;
  page?: number;
  section?: string;
  excerpt?: string;
  confidence?: number;
}

export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  citations?: Citation[];
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error?: string;
}

export interface DocumentState {
  documents: Document[];
  isUploading: boolean;
  uploadProgress: number;
  processingQueue: string[];
}