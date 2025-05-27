# Chat With Docs Frontend

## Setup

1. Install dependencies:

```bash
npm install
# or
yarn install
```

1. Set up environment variables:

```bash
cp .env.local.example .env.local
# Edit .env.local with your Clerk keys and API URL
```

1. Set up Clerk:

- Go to <https://clerk.dev> and create a new application
- Copy your publishable key and secret key to .env.local
- Configure allowed redirect URLs in Clerk dashboard

1. Run the development server:

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the
result.

## Building for Production

```bash
npm run build
npm start
```

## Deployment

The easiest way to deploy your Next.js app is to use the
[Vercel Platform](https://vercel.com/new).

Check out the [Next.js deployment documentation](https://nextjs.org/docs/deployment)
for more details.
