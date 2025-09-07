import '@testing-library/jest-dom'
import { vi, afterEach, beforeEach } from 'vitest'
import { setupGlobalMocks, resetAllMocks } from './utils'

// Setup global mocks before each test
beforeEach(() => {
  setupGlobalMocks()
})

// Mock DOM methods that aren't available in jsdom
Object.defineProperty(Element.prototype, 'scrollIntoView', {
  value: vi.fn(),
  writable: true,
})

// Setup cleanup after each test
afterEach(() => {
  resetAllMocks()
})
