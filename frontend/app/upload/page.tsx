'use client'

import React, { useState } from 'react'
import { DocumentUpload } from '../../src/components/DocumentUpload'
import { DocumentTable } from '../../src/components/DocumentTable'
import { useDocuments, useDocumentUpload, useDocumentDelete } from '../../src/hooks/useDocuments'
import { AlertCircle, CheckCircle } from 'lucide-react'

export default function UploadPage() {
  const { documents, isLoading, error, refetch } = useDocuments()
  const { uploadDocument, isUploading, uploadProgress, error: uploadError } = useDocumentUpload()
  const { deleteDocument, isDeleting, error: deleteError } = useDocumentDelete()
  
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null)

  const handleFileUpload = async (file: File) => {
    try {
      setUploadSuccess(null)
      const newDocument = await uploadDocument(file)
      
      // Simulate adding to the documents list (in real app, refetch would get this)
      setUploadSuccess(`Successfully uploaded: ${file.name}`)
      
      // Refetch documents to get updated list
      setTimeout(() => {
        refetch()
      }, 1000)
      
    } catch (error) {
      console.error('Upload failed:', error)
    }
  }

  const handleDeleteDocument = async (id: string) => {
    try {
      await deleteDocument(id)
      // Refetch documents to update the list
      refetch()
    } catch (error) {
      console.error('Delete failed:', error)
    }
  }

  return (
    <div className="min-h-screen px-6 py-6 space-y-6">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-100 mb-2">Document Library</h1>
        <p className="text-slate-400">Upload and manage your legal documents for analysis</p>
      </div>

      {/* Upload Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Document Upload */}
        <div className="glass-effect rounded-2xl p-4">
          <h2 className="text-xl font-semibold text-slate-200 mb-3">Upload Documents</h2>
          <DocumentUpload
            onUpload={handleFileUpload}
            isUploading={isUploading}
            uploadProgress={uploadProgress}
          />
          
          {/* Upload Status Messages */}
          <div className="mt-3 min-h-[50px]">
            {uploadError && (
              <div className="flex items-center space-x-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <span className="text-red-300 text-sm">{uploadError}</span>
              </div>
            )}
            
            {uploadSuccess && (
              <div className="flex items-center space-x-2 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
                <span className="text-green-300 text-sm">{uploadSuccess}</span>
              </div>
            )}
          </div>
        </div>

        {/* Upload Guidelines */}
        <div className="glass-effect rounded-2xl p-4">
          <h2 className="text-xl font-semibold text-slate-200 mb-3">Upload Guidelines</h2>
          <div className="space-y-3">
            <div className="space-y-2 text-sm text-slate-400">
              <div className="flex items-start space-x-3">
                <div className="w-1.5 h-1.5 bg-violet-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Supported formats: PDF, DOCX, TXT files</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-1.5 h-1.5 bg-violet-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Maximum file size: 10MB per document</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-1.5 h-1.5 bg-violet-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Documents are processed automatically after upload</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-1.5 h-1.5 bg-violet-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Processing time varies based on document size and complexity</span>
              </div>
            </div>
            
            <div className="pt-3 border-t border-slate-700/50">
              <div className="text-xs text-slate-500">
                <p>💡 <strong>Tip:</strong> For best results, ensure documents are text-searchable PDFs or properly formatted text files.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Documents List */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-slate-200">Your Documents</h2>
          {isLoading && (
            <div className="text-sm text-slate-400">Loading documents...</div>
          )}
        </div>

        {/* Error States */}
        {error && (
          <div className="flex items-center space-x-2 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <div className="text-red-300">
              <div className="font-medium">Failed to load documents</div>
              <div className="text-sm text-red-400">{error}</div>
            </div>
          </div>
        )}

        {deleteError && (
          <div className="flex items-center space-x-2 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <div className="text-red-300">
              <div className="font-medium">Failed to delete document</div>
              <div className="text-sm text-red-400">{deleteError}</div>
            </div>
          </div>
        )}

        {/* Document Table */}
        <DocumentTable
          documents={documents}
          onDelete={handleDeleteDocument}
          isDeleting={isDeleting}
        />
      </div>
    </div>
  )
} 