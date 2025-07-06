import React, { useState } from 'react'
import { FileText, Trash2, Clock, CheckCircle, AlertCircle } from 'lucide-react'
import { Document } from '../types'

interface DocumentTableProps {
  documents: Document[]
  onDelete: (id: string) => Promise<void>
  isDeleting?: string | null
}

export const DocumentTable: React.FC<DocumentTableProps> = ({
  documents,
  onDelete,
  isDeleting
}) => {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null)

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getStatusIcon = (status: Document['status'], progress?: number) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'processing':
        return <Clock className="w-4 h-4 text-amber-400" />
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-400" />
      default:
        return <Clock className="w-4 h-4 text-slate-400" />
    }
  }

  const getStatusText = (status: Document['status'], progress?: number) => {
    switch (status) {
      case 'ready':
        return 'Ready'
      case 'processing':
        return progress ? `Processing ${progress}%` : 'Processing'
      case 'failed':
        return 'Failed'
      default:
        return 'Unknown'
    }
  }

  const handleDeleteClick = (id: string) => {
    setShowDeleteConfirm(id)
  }

  const handleDeleteConfirm = async (id: string) => {
    try {
      await onDelete(id)
      setShowDeleteConfirm(null)
    } catch (error) {
      console.error('Delete failed:', error)
    }
  }

  const handleDeleteCancel = () => {
    setShowDeleteConfirm(null)
  }

  if (documents.length === 0) {
    return (
      <div className="glass-effect rounded-2xl p-8 text-center">
        <FileText className="w-12 h-12 text-slate-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-slate-300 mb-2">No documents uploaded yet</h3>
        <p className="text-slate-400">Upload your first document to get started</p>
      </div>
    )
  }

  return (
    <div className="glass-effect rounded-2xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-700/50 bg-slate-800/30">
              <th className="text-left px-6 py-4 text-sm font-medium text-slate-300">Document</th>
              <th className="text-left px-6 py-4 text-sm font-medium text-slate-300">Upload Date</th>
              <th className="text-left px-6 py-4 text-sm font-medium text-slate-300">Size</th>
              <th className="text-left px-6 py-4 text-sm font-medium text-slate-300">Status</th>
              <th className="text-right px-6 py-4 text-sm font-medium text-slate-300">Actions</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc, index) => (
              <tr
                key={doc.id}
                className={`border-b border-slate-700/30 hover:bg-slate-800/30 transition-colors ${
                  index === documents.length - 1 ? 'border-b-0' : ''
                }`}
              >
                <td className="px-6 py-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-violet-600 to-electric-600 rounded-lg flex items-center justify-center flex-shrink-0">
                      <FileText className="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <div className="font-medium text-slate-200 truncate max-w-xs">
                        {doc.filename}
                      </div>
                      {doc.pages && (
                        <div className="text-xs text-slate-400">
                          {doc.pages} page{doc.pages !== 1 ? 's' : ''}
                        </div>
                      )}
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 text-slate-300 text-sm">
                  {formatDate(doc.created_at)}
                </td>
                <td className="px-6 py-4 text-slate-300 text-sm">
                  {formatFileSize(doc.file_size)}
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(doc.status, doc.upload_progress)}
                    <span className="text-sm text-slate-300">
                      {getStatusText(doc.status, doc.upload_progress)}
                    </span>
                  </div>
                  {doc.status === 'processing' && doc.upload_progress && (
                    <div className="mt-1 w-32 bg-slate-700 rounded-full h-1.5">
                      <div
                        className="bg-gradient-to-r from-violet-600 to-electric-600 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${doc.upload_progress}%` }}
                      />
                    </div>
                  )}
                </td>
                <td className="px-6 py-4 text-right">
                  {showDeleteConfirm === doc.id ? (
                    <div className="flex items-center justify-end space-x-2">
                      <button
                        onClick={() => handleDeleteConfirm(doc.id)}
                        disabled={isDeleting === doc.id}
                        className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded-lg transition-colors disabled:opacity-50"
                      >
                        {isDeleting === doc.id ? 'Deleting...' : 'Confirm'}
                      </button>
                      <button
                        onClick={handleDeleteCancel}
                        className="px-3 py-1 bg-slate-600 hover:bg-slate-700 text-white text-xs rounded-lg transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => handleDeleteClick(doc.id)}
                      className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                      title="Delete document"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
} 