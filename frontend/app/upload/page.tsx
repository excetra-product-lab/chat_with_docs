'use client'

import React, { useState } from 'react'
import { useAuth } from '@clerk/nextjs'
import { DocumentUpload } from '../../src/components/DocumentUpload'
import { DocumentTable } from '../../src/components/DocumentTable'
import { useDocuments, useDocumentUpload, useDocumentDelete } from '../../src/hooks/useDocuments'
import { DocumentErrorBoundary } from '../../src/components/ErrorBoundary'
import { useToastWithErrorHandling } from '../../src/components/Toast'
import { AlertCircle, CheckCircle, UserCheck } from 'lucide-react'

export default function UploadPage() {
  const { isSignedIn, isLoaded } = useAuth()
  const { documents, isLoading, error, refetch } = useDocuments()
  const { uploadDocument, isUploading, uploadProgress, error: uploadError } = useDocumentUpload()
  const { deleteDocument, isDeleting, error: deleteError } = useDocumentDelete()
  const toast = useToastWithErrorHandling()

  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null)

  const handleFileUpload = async (file: File) => {
    try {
      setUploadSuccess(null)
      const newDocument = await uploadDocument(file)

      toast.showSuccess(
        'Document uploaded successfully',
        `${file.name} is being processed and will be available for chat soon.`
      )

      // Refetch documents to get updated list
      setTimeout(() => {
        refetch()
      }, 2000)

    } catch (error) {
      console.error('Upload failed:', error)
      toast.showApiError(error, 'Upload')
    }
  }

  const handleDeleteDocument = async (id: number) => {
    try {
      await deleteDocument(id)
      // Refetch documents to update the list
      refetch()
      
      toast.showSuccess(
        'Document deleted',
        'The document has been successfully removed.'
      )
    } catch (error) {
      console.error('Delete failed:', error)
      toast.showApiError(error, 'Delete')
    }
  }

  // Show loading state while Clerk is initializing
  if (!isLoaded) {
    return (
      <div className="min-h-screen px-6 py-6 flex items-center justify-center">
        <div className="text-slate-400">Loading...</div>
      </div>
    )
  }

  // Show sign-in prompt for unauthenticated users
  if (!isSignedIn) {
    return (
      <div className="min-h-screen px-6 py-6 flex items-center justify-center">
        <div className="max-w-md mx-auto text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-violet-600 to-electric-600 rounded-full flex items-center justify-center mx-auto mb-6">
            <UserCheck className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-slate-100 mb-4">Sign In Required</h1>
          <p className="text-slate-400 mb-8">
            Please sign in to upload and manage your legal documents. Your documents are securely stored and only accessible to you.
          </p>
          <div className="space-y-4">
            <div className="text-sm text-slate-500">
              <p>✓ Secure document storage</p>
              <p>✓ AI-powered analysis</p>
              <p>✓ Precise citations</p>
            </div>
          </div>
        </div>
      </div>
    )
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
        <DocumentErrorBoundary>
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
        </DocumentErrorBoundary>

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
        <DocumentErrorBoundary>
          <DocumentTable
            documents={documents}
            onDelete={handleDeleteDocument}
            isDeleting={isDeleting}
            isLoading={isLoading}
          />
        </DocumentErrorBoundary>
      </div>
    </div>
  )
}
