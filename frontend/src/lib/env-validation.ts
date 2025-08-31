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

  // Check required environment variables
  for (const envVar of REQUIRED_ENV_VARS) {
    const value = process.env[envVar]
    
    if (!value) {
      errors.push(`Missing required environment variable: ${envVar}`)
    } else if (value.includes('your-') || value.includes('here')) {
      errors.push(`Environment variable ${envVar} appears to contain placeholder value: ${value}`)
    }
  }

  // Check optional environment variables for warnings
  for (const envVar of OPTIONAL_ENV_VARS) {
    const value = process.env[envVar]
    
    if (!value) {
      warnings.push(`Optional environment variable not set: ${envVar}`)
    }
  }

  // Validate specific environment variable formats
  validateApiUrl(errors)
  validateClerkKeys(errors)

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  }
}

/**
 * Validates API URL format
 */
function validateApiUrl(errors: string[]) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL
  
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
function validateClerkKeys(errors: string[]) {
  const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
  
  if (publishableKey) {
    if (!publishableKey.startsWith('pk_')) {
      errors.push(`NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY should start with 'pk_': ${publishableKey}`)
    }
  }
}

/**
 * Gets the validated environment configuration
 * Throws an error if validation fails
 */
export function getEnvironmentConfig(): EnvironmentConfig {
  const validation = validateEnvironment()
  
  if (!validation.isValid) {
    const errorMessage = [
      '❌ Environment validation failed!',
      '',
      'Missing or invalid environment variables:',
      ...validation.errors.map(error => `  • ${error}`),
      '',
      'Please check your .env.local file and ensure all required variables are set.',
      'See the README.md for setup instructions.'
    ].join('\n')
    
    throw new Error(errorMessage)
  }

  // Log warnings in development
  if (process.env.NODE_ENV === 'development' && validation.warnings.length > 0) {
    console.warn('⚠️ Environment warnings:')
    validation.warnings.forEach(warning => console.warn(`  • ${warning}`))
  }

  return {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL!,
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY!
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

