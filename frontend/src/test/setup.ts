import '@testing-library/jest-dom'
import { vi, afterEach } from 'vitest'

// Mock DOM methods that aren't available in jsdom
Object.defineProperty(Element.prototype, 'scrollIntoView', {
  value: vi.fn(),
  writable: true,
})

// Setup cleanup after each test
afterEach(() => {
  vi.clearAllMocks()
})
