import React, { useCallback, useState } from 'react';
import { Upload, FileText, AlertCircle, Plus } from 'lucide-react';

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
    <div>
      <div
        className={`text-center transition-all duration-200 ${
          isDragOver ? 'scale-105' : ''
        } ${isUploading ? 'pointer-events-none' : 'cursor-pointer'}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && document.getElementById('file-upload')?.click()}
      >
        {isUploading ? (
          <div className="space-y-3">
            <div className="w-10 h-10 mx-auto relative">
              <div className="absolute inset-0 rounded-full border-2 border-slate-700"></div>
              <div className="absolute inset-0 rounded-full border-2 border-violet-500 border-t-transparent animate-spin"></div>
            </div>
            <div className="space-y-1">
              <p className="text-slate-200 font-medium">Uploading document</p>
              <p className="text-slate-400 text-sm">{uploadProgress}% complete</p>
            </div>
            <div className="max-w-xs mx-auto">
              <div className="w-full bg-slate-700/50 rounded-full h-1.5">
                <div
                  className="bg-violet-500 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="w-12 h-12 mx-auto flex items-center justify-center">
              <Upload className="w-6 h-6 text-slate-400" />
            </div>

            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-slate-200">
                Upload your legal documents
              </h3>
              <p className="text-slate-400 text-sm max-w-sm mx-auto">
                Drag and drop files here, or click to browse your computer
              </p>
            </div>

            <label htmlFor="file-upload" className="cursor-pointer inline-block">
              <div className="inline-flex items-center px-4 py-2 bg-slate-700/50 hover:bg-slate-600/50 text-slate-200 rounded-lg transition-colors font-medium text-sm">
                <Plus className="w-4 h-4 mr-2" />
                Choose Files
              </div>
            </label>

            <input
              id="file-upload"
              type="file"
              className="hidden"
              accept=".pdf,.docx,.txt"
              onChange={handleFileSelect}
            />

            <div className="flex items-center justify-center space-x-4 text-xs text-slate-500">
              <span className="flex items-center">
                <FileText className="w-3 h-3 mr-1" />
                PDF, DOCX, TXT
              </span>
              <span>Max 10MB</span>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <span className="text-red-300 text-sm">{error}</span>
          </div>
        </div>
      )}
    </div>
  );
};
