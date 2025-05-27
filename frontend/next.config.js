const path = require('path');

/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  experimental: {
    typedRoutes: true,
  },
  webpack: (config) => {
    // Add comprehensive path aliases
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': path.resolve(__dirname, '.'),
      '@/lib': path.resolve(__dirname, 'lib'),
      '@/components': path.resolve(__dirname, 'components'),
      '@/app': path.resolve(__dirname, 'app'),
      '@/styles': path.resolve(__dirname, 'styles'),
      '@/context': path.resolve(__dirname, 'context'),
      '@/public': path.resolve(__dirname, 'public'),
    };

    // Ensure TypeScript paths are resolved
    config.resolve.extensions = ['.ts', '.tsx', '.js', '.jsx', '.json'];

    return config;
  },
};

module.exports = nextConfig;
