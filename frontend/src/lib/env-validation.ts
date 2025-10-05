/**
 * Environment variable validation utility
 * Validates required environment variables at application startup
 * and provides clear error messages for missing configuration
 */

interface EnvironmentConfig {
  NEXT_PUBLIC_API_URL: string
  NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: string
}

interface ValidationResult {
  isValid: boolean
  errors: string[]
  warnings: string[]
}

/**
 * Required environment variables for the application to function
 */
const REQUIRED_ENV_VARS = [
  'NEXT_PUBLIC_API_URL',
  'NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY'
] as const

/**
 * Optional environment variables that enhance functionality
 */
const OPTIONAL_ENV_VARS = [
  'CLERK_SECRET_KEY',
  'CLERK_SIGN_IN_URL',
  'CLERK_SIGN_UP_URL',
  'CLERK_AFTER_SIGN_IN_URL',
  'CLERK_AFTER_SIGN_UP_URL'
] as const

/**
 * Validates all required environment variables
 * @returns ValidationResult with errors and warnings
 */
export function validateEnvironment(): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []

  // Check required environment variables - access directly by name for Next.js client-side compatibility
  const NEXT_PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL
  const NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
  
  // Validate NEXT_PUBLIC_API_URL
  if (!NEXT_PUBLIC_API_URL) {
    errors.push(`Missing required environment variable: NEXT_PUBLIC_API_URL`)
  } else if (NEXT_PUBLIC_API_URL.includes('your-') || NEXT_PUBLIC_API_URL.includes('here')) {
    errors.push(`Environment variable NEXT_PUBLIC_API_URL appears to contain placeholder value: ${NEXT_PUBLIC_API_URL}`)
  }
  
  // Validate NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
  if (!NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY) {
    errors.push(`Missing required environment variable: NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`)
  } else if (NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY.includes('your-') || NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY.includes('here')) {
    errors.push(`Environment variable NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY appears to contain placeholder value: ${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}`)
  }

  // Check optional environment variables for warnings
  for (const envVar of OPTIONAL_ENV_VARS) {
    const value = process.env[envVar]
    
    if (!value) {
      warnings.push(`Optional environment variable not set: ${envVar}`)
    }
  }

  // Validate specific environment variable formats
  validateApiUrl(errors, NEXT_PUBLIC_API_URL)
  validateClerkKeys(errors, NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY)

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  }
}

/**
 * Validates API URL format
 */
function validateApiUrl(errors: string[], apiUrl: string | undefined) {
  if (apiUrl) {
    try {
      new URL(apiUrl)
    } catch {
      errors.push(`NEXT_PUBLIC_API_URL is not a valid URL: ${apiUrl}`)
    }
    
    if (apiUrl.endsWith('/')) {
      errors.push(`NEXT_PUBLIC_API_URL should not end with a slash: ${apiUrl}`)
    }
  }
}

/**
 * Validates Clerk key formats
 */
function validateClerkKeys(errors: string[], publishableKey: string | undefined) {
  if (publishableKey) {
    if (!publishableKey.startsWith('pk_')) {
      errors.push(`NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY should start with 'pk_': ${publishableKey}`)
    }
  }
}

/**
 * Gets the validated environment configuration
 * Always returns usable values with fallbacks
 */
export function getEnvironmentConfig(): EnvironmentConfig {
  // Always return usable configuration with fallbacks
  return {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || ''
  }
}

/**
 * Utility to check if we're in development mode
 */
export function isDevelopment(): boolean {
  return process.env.NODE_ENV === 'development'
}

/**
 * Utility to get the current environment
 */
export function getEnvironment(): 'development' | 'production' | 'test' {
  return (process.env.NODE_ENV as any) || 'development'
}

