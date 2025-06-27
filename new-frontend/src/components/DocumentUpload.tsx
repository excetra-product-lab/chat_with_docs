import React, { useCallback, useState } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';

interface DocumentUploadProps {
  onUpload: (file: File) => void;
  isUploading: boolean;
  uploadProgress: number;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUpload,
  isUploading,
  uploadProgress
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const validateFile = (file: File): string | null => {
    const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
    if (!allowedTypes.includes(file.type)) {
      return 'Only PDF, DOCX, and TXT files are supported';
    }
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      return 'File size must be less than 10MB';
    }
    return null;
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    setError(null);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      const file = files[0];
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }
      onUpload(file);
    }
  }, [onUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }
      onUpload(file);
    }
  }, [onUpload]);

  return (
    <div className="p-4 border-t border-slate-800/50 bg-gradient-to-r from-midnight-950/80 to-slate-900/80 backdrop-blur-sm">
      <div
        className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 ${
          isDragOver
            ? 'border-violet-400 bg-violet-400/10 shadow-glow-sm'
            : 'border-slate-600 hover:border-violet-500/50'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isUploading ? (
          <div className="space-y-3">
            <div className="w-8 h-8 mx-auto">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
            </div>
            <p className="text-slate-300 font-light">Uploading... {uploadProgress}%</p>
            <div className="w-full bg-slate-700/50 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-violet-500 to-electric-500 h-2 rounded-full transition-all duration-300 shadow-glow-sm"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        ) : (
          <>
            <Upload className="w-8 h-8 mx-auto mb-3 text-slate-400" />
            <p className="text-slate-300 mb-2 font-light">Drop documents here or</p>
            <label htmlFor="file-upload" className="cursor-pointer">
              <span className="button-primary inline-flex items-center">
                <FileText className="w-4 h-4 mr-2" />
                Browse Files
              </span>
            </label>
            <input
              id="file-upload"
              type="file"
              className="hidden"
              accept=".pdf,.docx,.txt"
              onChange={handleFileSelect}
            />
            <p className="text-xs text-slate-500 mt-2 font-light">
              Supports PDF, DOCX, TXT (max 10MB)
            </p>
          </>
        )}
      </div>
      
      {error && (
        <div className="mt-3 p-3 bg-red-900/30 border border-red-600/50 rounded-xl flex items-center backdrop-blur-sm">
          <AlertCircle className="w-4 h-4 text-red-400 mr-2" />
          <span className="text-red-300 text-sm font-light">{error}</span>
        </div>
      )}
    </div>
  );
};