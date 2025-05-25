import { UserButton } from '@clerk/nextjs'
import Link from 'next/link'

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white shadow-sm border-b">
        <div className="container mx-auto px-4 py-3 flex justify-between items-center">
          <Link href="/" className="text-xl font-bold">
            Chat With Docs
          </Link>
          <nav className="flex items-center gap-6">
            <Link href="/upload" className="hover:text-blue-600">
              Documents
            </Link>
            <Link href="/chat" className="hover:text-blue-600">
              Chat
            </Link>
            <UserButton afterSignOutUrl="/" />
          </nav>
        </div>
      </header>
      <main className="flex-1 bg-gray-50">
        {children}
      </main>
    </div>
  )
}
