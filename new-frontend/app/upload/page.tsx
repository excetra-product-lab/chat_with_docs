'use client'

import { DocumentLibrary } from '../../src/components/DocumentLibrary'
import { useAppContext } from '../../src/context/AppContext'

export default function UploadPage() {
  const { documents, addDocument, updateDocument, deleteDocument, isLoading, setIsLoading } = useAppContext()

  const uploadDocument = async (file: File) => {
    // Mock upload logic - in real app this would call API
    const newDocument = {
      id: Date.now().toString(),
      name: file.name,
      size: file.size,
      type: file.type.includes('pdf') ? 'pdf' as const : 
            file.type.includes('doc') ? 'docx' as const : 'txt' as const,
      status: 'processing' as const,
      uploadProgress: 0,
      uploadDate: new Date(),
      pages: Math.floor(file.size / 2000), // Mock page count
    }
    
    addDocument(newDocument)
    setIsLoading(true)
    
    // Simulate upload progress
    for (let progress = 0; progress <= 100; progress += 10) {
      await new Promise(resolve => setTimeout(resolve, 100))
      updateDocument(newDocument.id, { uploadProgress: progress })
    }
    
    // Mark as ready
    updateDocument(newDocument.id, { 
      status: 'ready', 
      uploadProgress: 100
    })
    setIsLoading(false)
  }

  return (
    <div className="h-screen px-6">
      <DocumentLibrary
        documents={documents}
        onUpload={uploadDocument}
        onDelete={deleteDocument}
        isUploading={isLoading}
        uploadProgress={100}
        isCollapsed={false}
        onToggleCollapse={() => {}}
      />
    </div>
  )
} 