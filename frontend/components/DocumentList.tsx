'use client'

import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '@clerk/nextjs'
import { api } from '@/lib/api'
import { Trash2, FileText } from 'lucide-react'

interface Document {
  id: number
  filename: string
  status: string
  created_at: string
}

export default function DocumentList() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const { getToken } = useAuth()

  const fetchDocuments = useCallback(async () => {
    try {
      const token = await getToken()
      const response = await api.getDocuments(token!)
      setDocuments(response.data)
    } catch (error) {
      console.error('Error fetching documents:', error)
    } finally {
      setLoading(false)
    }
  }, [getToken])

  useEffect(() => {
    fetchDocuments()
  }, [fetchDocuments])

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this document?')) return

    try {
      const token = await getToken()
      await api.deleteDocument(id, token!)
      await fetchDocuments()
    } catch (error) {
      console.error('Error deleting document:', error)
    }
  }

  if (loading) return <div>Loading documents...</div>

  if (documents.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No documents uploaded yet
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className="flex items-center justify-between p-3 bg-white rounded-lg shadow-sm border"
        >
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-gray-600" />
            <div>
              <p className="font-medium">{doc.filename}</p>
              <p className="text-sm text-gray-500">
                Status: {doc.status} • {new Date(doc.created_at).toLocaleDateString()}
              </p>
            </div>
          </div>
          <button
            onClick={() => handleDelete(doc.id)}
            className="p-2 text-red-600 hover:bg-red-50 rounded"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      ))}
    </div>
  )
}
