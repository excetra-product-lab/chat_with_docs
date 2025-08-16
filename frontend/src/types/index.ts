export interface Document {
  id: number;  // Changed from string to number to match backend
  filename: string;
  user_id: number;  // Changed from string to number to match backend
  status: 'processing' | 'ready' | 'failed';
  created_at: string;
  file_size: number;
  file_type: string;
  pages?: number;
  upload_progress?: number;
}

export interface DocumentChunk {
  id: number;  // Changed from string to number to match backend
  documentId: number;  // Changed from string to number to match backend
  content: string;
  page?: number;
  section?: string;
  startIndex: number;
  endIndex: number;
}

export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
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
