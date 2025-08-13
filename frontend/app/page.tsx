'use client'

import React, { useState } from 'react'
import { useRouter } from 'next/navigation'
import { LandingPage } from '../src/components/LandingPage'

export default function HomePage() {
  const router = useRouter()

  const handleGetStarted = () => {
    router.push('/upload')
  }

  return (
    <main>
      <LandingPage onGetStarted={handleGetStarted} />
    </main>
  )
}
