'use client'

import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useAuth } from '@clerk/nextjs'
import { api } from '@/lib/api'
import { Upload, File } from 'lucide-react'

export default function UploadForm() {
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')
  const { getToken } = useAuth()

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return

    setUploading(true)
    setMessage('')

    try {
      const token = await getToken()
      const file = acceptedFiles[0]
      
      await api.uploadDocument(file, token!)
      setMessage('Document uploaded successfully!')
      
      // Refresh document list (you might want to use a global state or callback here)
      window.location.reload()
    } catch (error) {
      setMessage('Error uploading document')
      console.error('Upload error:', error)
    } finally {
      setUploading(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    maxFiles: 1
  })

  return (
    <div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
      >
        <input {...getInputProps()} />
        <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        {isDragActive ? (
          <p className="text-blue-600">Drop the file here...</p>
        ) : (
          <div>
            <p className="text-gray-600 mb-2">
              Drag and drop a document here, or click to select
            </p>
            <p className="text-sm text-gray-500">
              Supported formats: PDF, DOCX, TXT
            </p>
          </div>
        )}
      </div>
      
      {uploading && (
        <p className="mt-4 text-blue-600">Uploading...</p>
      )}
      
      {message && (
        <p className={`mt-4 ${message.includes('Error') ? 'text-red-600' : 'text-green-600'}`}>
          {message}
        </p>
      )}
    </div>
  )
}
