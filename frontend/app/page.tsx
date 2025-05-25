import { SignInButton, SignedIn, SignedOut, UserButton } from '@clerk/nextjs'
import Link from 'next/link'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold mb-8">Chat With Docs</h1>
      </div>

      <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-1 lg:text-left">
        <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100">
          <h2 className="mb-3 text-2xl font-semibold">
            AI-Powered Document Q&A
          </h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Upload your legal documents and ask questions. Get accurate answers with citations.
          </p>
          
          <div className="mt-6">
            <SignedOut>
              <SignInButton mode="modal">
                <button className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600">
                  Get Started
                </button>
              </SignInButton>
            </SignedOut>
            <SignedIn>
              <div className="flex gap-4">
                <Link href="/upload" className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600">
                  Upload Documents
                </Link>
                <Link href="/chat" className="bg-green-500 text-white px-6 py-2 rounded-md hover:bg-green-600">
                  Start Chatting
                </Link>
                <UserButton afterSignOutUrl="/" />
              </div>
            </SignedIn>
          </div>
        </div>
      </div>
    </main>
  )
}
