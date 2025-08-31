import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  validateEnvironment,
  getEnvironmentConfig,
  isDevelopment,
  getEnvironment
} from '../lib/env-validation'

describe('env-validation', () => {
  const originalEnv = process.env

  beforeEach(() => {
    vi.clearAllMocks()
    // Clear all environment variables
    Object.keys(process.env).forEach(key => {
      delete (process.env as any)[key]
    })
  })

  afterEach(() => {
    process.env = originalEnv
  })

  describe('validateEnvironment', () => {
    it('should pass validation with all required variables set', () => {
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_test_abcd1234'

      const result = validateEnvironment()

      expect(result.isValid).toBe(true)
      expect(result.errors).toEqual([])
      expect(result.warnings.length).toBeGreaterThan(0) // Optional vars will have warnings
    })

    it('should fail validation with missing required variables', () => {
      const result = validateEnvironment()

      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Missing required environment variable: NEXT_PUBLIC_API_URL')
      expect(result.errors).toContain('Missing required environment variable: NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY')
    })

    it('should detect placeholder values', () => {
      process.env.NEXT_PUBLIC_API_URL = 'your-api-url-here'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_test_your-clerk-key-here'

      const result = validateEnvironment()

      expect(result.isValid).toBe(false)
      expect(result.errors).toContain(
        'Environment variable NEXT_PUBLIC_API_URL appears to contain placeholder value: your-api-url-here'
      )
      expect(result.errors).toContain(
        'Environment variable NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY appears to contain placeholder value: pk_test_your-clerk-key-here'
      )
    })

    it('should generate warnings for missing optional variables', () => {
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_test_abcd1234'

      const result = validateEnvironment()

      expect(result.isValid).toBe(true)
      expect(result.warnings).toContain('Optional environment variable not set: CLERK_SECRET_KEY')
      expect(result.warnings).toContain('Optional environment variable not set: CLERK_SIGN_IN_URL')
      expect(result.warnings).toContain('Optional environment variable not set: CLERK_SIGN_UP_URL')
      expect(result.warnings).toContain('Optional environment variable not set: CLERK_AFTER_SIGN_IN_URL')
      expect(result.warnings).toContain('Optional environment variable not set: CLERK_AFTER_SIGN_UP_URL')
    })

    it('should not warn for set optional variables', () => {
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_test_abcd1234'
      process.env.CLERK_SECRET_KEY = 'sk_test_secret'
      process.env.CLERK_SIGN_IN_URL = '/sign-in'

      const result = validateEnvironment()

      expect(result.warnings).not.toContain('Optional environment variable not set: CLERK_SECRET_KEY')
      expect(result.warnings).not.toContain('Optional environment variable not set: CLERK_SIGN_IN_URL')
    })
  })

  describe('API URL validation', () => {
    beforeEach(() => {
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_test_abcd1234'
    })

    it('should accept valid URLs', () => {
      const validUrls = [
        'http://localhost:8000',
        'https://api.example.com',
        'https://api.example.com:3000',
        'http://127.0.0.1:8000'
      ]

      validUrls.forEach(url => {
        process.env.NEXT_PUBLIC_API_URL = url
        const result = validateEnvironment()
        expect(result.isValid).toBe(true)
      })
    })

    it('should reject invalid URLs', () => {
      const invalidUrls = [
        'not-a-url',
        'just-a-string',
        '://malformed',
        ''
      ]

      invalidUrls.forEach(url => {
        process.env.NEXT_PUBLIC_API_URL = url
        const result = validateEnvironment()
        if (url === '') {
          // Empty URL should fail validation for missing required var
          expect(result.isValid).toBe(false)
          expect(result.errors.some(error => 
            error.includes('Missing required environment variable')
          )).toBe(true)
        } else {
          expect(result.isValid).toBe(false)
          expect(result.errors.some(error => 
            error.includes('NEXT_PUBLIC_API_URL is not a valid URL')
          )).toBe(true)
        }
      })
    })

    it('should reject URLs ending with slash', () => {
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000/'

      const result = validateEnvironment()

      expect(result.isValid).toBe(false)
      expect(result.errors).toContain(
        'NEXT_PUBLIC_API_URL should not end with a slash: http://localhost:8000/'
      )
    })
  })

  describe('Clerk key validation', () => {
    beforeEach(() => {
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000'
    })

    it('should accept valid Clerk publishable keys', () => {
      const validKeys = [
        'pk_test_abcd1234',
        'pk_live_xyz789',
        'pk_test_very-long-key-with-many-characters-123456789'
      ]

      validKeys.forEach(key => {
        process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = key
        const result = validateEnvironment()
        expect(result.isValid).toBe(true)
      })
    })

    it('should reject invalid Clerk publishable keys', () => {
      const invalidKeys = [
        'sk_test_secret',
        'abcd1234',
        'clerk_key_123'
      ]

      invalidKeys.forEach(key => {
        process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = key
        const result = validateEnvironment()
        expect(result.isValid).toBe(false)
        expect(result.errors.some(error => 
          error.includes("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY should start with 'pk_'")
        )).toBe(true)
      })
    })
  })

  describe('getEnvironmentConfig', () => {
    it('should return validated configuration', () => {
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_test_abcd1234'

      const config = getEnvironmentConfig()

      expect(config).toEqual({
        NEXT_PUBLIC_API_URL: 'http://localhost:8000',
        NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: 'pk_test_abcd1234'
      })
    })

    it('should throw error when validation fails', () => {
      process.env.NEXT_PUBLIC_API_URL = 'invalid-url'

      expect(() => getEnvironmentConfig()).toThrow(/Environment validation failed/)
    })

    it('should include error details in thrown error', () => {
      expect(() => getEnvironmentConfig()).toThrow(/Missing or invalid environment variables/)
    })

    it('should log warnings in development mode', () => {
      (process.env as any).NODE_ENV = 'development'
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_test_abcd1234'

      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

      getEnvironmentConfig()

      expect(consoleSpy).toHaveBeenCalledWith('⚠️ Environment warnings:')
      expect(consoleSpy).toHaveBeenCalledWith('  • Optional environment variable not set: CLERK_SECRET_KEY')

      consoleSpy.mockRestore()
    })

    it('should not log warnings in production mode', () => {
      (process.env as any).NODE_ENV = 'production'
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_test_abcd1234'

      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

      getEnvironmentConfig()

      expect(consoleSpy).not.toHaveBeenCalled()

      consoleSpy.mockRestore()
    })
  })

  describe('isDevelopment', () => {
    it('should return true in development environment', () => {
      (process.env as any).NODE_ENV = 'development'
      expect(isDevelopment()).toBe(true)
    })

    it('should return false in production environment', () => {
      (process.env as any).NODE_ENV = 'production'
      expect(isDevelopment()).toBe(false)
    })

    it('should return false in test environment', () => {
      (process.env as any).NODE_ENV = 'test'
      expect(isDevelopment()).toBe(false)
    })

    it('should return false when NODE_ENV is not set', () => {
      delete (process.env as any).NODE_ENV
      expect(isDevelopment()).toBe(false)
    })
  })

  describe('getEnvironment', () => {
    it('should return development environment', () => {
      (process.env as any).NODE_ENV = 'development'
      const result = getEnvironment()
      expect(result).toBe('development')
    })

    it('should return production environment', () => {
      (process.env as any).NODE_ENV = 'production'
      const result = getEnvironment()
      expect(result).toBe('production')
    })

    it('should return test environment', () => {
      (process.env as any).NODE_ENV = 'test'
      const result = getEnvironment()
      expect(result).toBe('test')
    })

    it('should default to development when NODE_ENV is not set', () => {
      delete (process.env as any).NODE_ENV
      const result = getEnvironment()
      expect(result).toBe('development')
    })

    it('should handle custom environment values', () => {
      (process.env as any).NODE_ENV = 'staging' as any
      const result = getEnvironment()
      expect(result).toBe('staging')
    })
  })

  describe('integration scenarios', () => {
    it('should handle complete valid configuration', () => {
      process.env.NEXT_PUBLIC_API_URL = 'https://api.example.com'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_live_production_key'
      process.env.CLERK_SECRET_KEY = 'sk_live_secret'
      process.env.CLERK_SIGN_IN_URL = '/sign-in'
      process.env.CLERK_SIGN_UP_URL = '/sign-up'
      process.env.CLERK_AFTER_SIGN_IN_URL = '/dashboard'
      process.env.CLERK_AFTER_SIGN_UP_URL = '/onboarding'

      const validation = validateEnvironment()
      expect(validation.isValid).toBe(true)
      expect(validation.errors).toEqual([])
      expect(validation.warnings).toEqual([])

      const config = getEnvironmentConfig()
      expect(config.NEXT_PUBLIC_API_URL).toBe('https://api.example.com')
      expect(config.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY).toBe('pk_live_production_key')
    })

    it('should handle minimal valid configuration with warnings', () => {
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:3000'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'pk_test_dev_key'

      const validation = validateEnvironment()
      expect(validation.isValid).toBe(true)
      expect(validation.errors).toEqual([])
      expect(validation.warnings.length).toBe(5) // All optional vars missing

      const config = getEnvironmentConfig()
      expect(config).toBeDefined()
    })

    it('should handle multiple validation errors', () => {
      process.env.NEXT_PUBLIC_API_URL = 'invalid-url/'
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = 'invalid-key'

      const validation = validateEnvironment()
      expect(validation.isValid).toBe(false)
      expect(validation.errors.length).toBeGreaterThan(1)
      expect(validation.errors.some(error => error.includes('not a valid URL'))).toBe(true)
      expect(validation.errors.some(error => error.includes('should not end with a slash'))).toBe(true)
      expect(validation.errors.some(error => error.includes("should start with 'pk_'"))).toBe(true)
    })
  })
})
