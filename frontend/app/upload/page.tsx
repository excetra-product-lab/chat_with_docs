'use client'

import { useAuth } from '@clerk/nextjs'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import Layout from '@/components/Layout'
import UploadForm from '@/components/UploadForm'
import DocumentList from '@/components/DocumentList'

export default function UploadPage() {
  const { isLoaded, isSignedIn } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (isLoaded && !isSignedIn) {
      router.push('/')
    }
  }, [isLoaded, isSignedIn, router])

  if (!isLoaded || !isSignedIn) {
    return <div>Loading...</div>
  }

  return (
    <Layout>
      <div className="container mx-auto p-4">
        <h1 className="text-2xl font-bold mb-4">Document Management</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h2 className="text-xl font-semibold mb-3">Upload New Document</h2>
            <UploadForm />
          </div>
          <div>
            <h2 className="text-xl font-semibold mb-3">Your Documents</h2>
            <DocumentList />
          </div>
        </div>
      </div>
    </Layout>
  )
}
