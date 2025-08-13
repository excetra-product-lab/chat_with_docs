import React from 'react';
import { FileText, File, Trash2, CheckCircle, Clock, AlertCircle, ChevronLeft, ChevronRight, FolderOpen } from 'lucide-react';
import { Document } from '../types';
import { DocumentUpload } from './DocumentUpload';

interface DocumentLibraryProps {
  documents: Document[];
  onUpload: (file: File) => void;
  onDelete: (id: string) => void;
  isUploading: boolean;
  uploadProgress: number;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export const DocumentLibrary: React.FC<DocumentLibraryProps> = ({
  documents,
  onUpload,
  onDelete,
  isUploading,
  uploadProgress,
  isCollapsed,
  onToggleCollapse
}) => {
  const getDocumentIcon = (type: string) => {
    if (type.includes('pdf')) {
      return <FileText className="w-5 h-5 text-red-400" />;
    } else if (type.includes('word') || type.includes('document')) {
      return <File className="w-5 h-5 text-blue-400" />;
    } else {
      return <File className="w-5 h-5 text-slate-400" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'processing':
        return (
          <div className="w-4 h-4">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-violet-400"></div>
          </div>
        );
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-slate-400" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  if (isCollapsed) {
    return (
      <div className="flex flex-col h-full bg-gradient-to-br from-midnight-950 to-slate-900">
        <div className="px-4 py-6 border-b border-slate-800/50 bg-gradient-to-r from-midnight-950/80 to-slate-900/80 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <FolderOpen className="w-6 h-6 text-violet-400" />
            <button
              onClick={onToggleCollapse}
              className="p-1 text-slate-400 hover:text-slate-200 transition-colors"
              title="Expand library"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 flex flex-col items-center justify-center p-4">
          <div className="text-center space-y-4">
            <div className="text-2xl font-bold text-violet-400">{documents.length}</div>
            <div className="text-xs text-slate-400 writing-mode-vertical transform rotate-180">
              Documents
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-midnight-950 to-slate-900">
      <div className="px-6 py-6 border-b border-slate-800/50 bg-gradient-to-r from-midnight-950/80 to-slate-900/80 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-slate-100 mb-1 tracking-tight">Document Library</h2>
            <p className="text-slate-400 text-sm font-light">Upload and manage your legal documents</p>
          </div>
          <button
            onClick={onToggleCollapse}
            className="p-2 text-slate-400 hover:text-slate-200 transition-colors"
            title="Collapse library"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-3">
          {documents.map((document) => (
            <div
              key={document.id}
              className="glass-effect rounded-xl p-4 hover:border-violet-500/50 transition-all duration-300 group"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1 min-w-0">
                  {getDocumentIcon(document.file_type)}
                  <div className="flex-1 min-w-0">
                    <h3 className="text-slate-100 font-medium truncate" title={document.filename}>
                      {document.filename}
                    </h3>
                    <div className="flex items-center space-x-4 text-xs text-slate-400 mt-1 font-light">
                      <span>{formatFileSize(document.file_size)}</span>
                      <span>{formatDate(new Date(document.created_at))}</span>
                      {document.pages && <span>{document.pages} pages</span>}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {getStatusIcon(document.status)}
                  {document.status === 'ready' && (
                    <button
                      onClick={() => onDelete(document.id)}
                      className="p-1 text-slate-500 hover:text-red-400 transition-colors"
                      title="Delete document"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>

              <div className="flex items-center mt-2">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${
                    document.status === 'ready' ? 'bg-emerald-400' :
                    document.status === 'processing' ? 'bg-violet-400' :
                    'bg-red-400'
                  }`}></div>
                  <span className={`text-xs capitalize ${
                    document.status === 'ready' ? 'text-emerald-400' :
                    document.status === 'processing' ? 'text-violet-400' :
                    'text-red-400'
                  }`}>
                    {document.status}
                  </span>
                </div>
              </div>
            </div>
          ))}

          {documents.length === 0 && (
            <div className="text-center py-8">
              <FileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400 font-light">No documents uploaded yet</p>
              <p className="text-slate-500 text-sm font-light">Upload your first document to get started</p>
            </div>
          )}
        </div>
      </div>

      <DocumentUpload
        onUpload={onUpload}
        isUploading={isUploading}
        uploadProgress={uploadProgress}
      />
    </div>
  );
};
