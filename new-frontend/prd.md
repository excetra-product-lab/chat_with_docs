# Product Requirements Document (PRD)
## EXCETRA-14: Chat UI and Document UI (Frontend)

### Project Overview
**Linear Issue**: EXCETRA-14  
**Status**: In Progress  
**Assignee**: pauldoho@outlook.com  
**Target Audience**: Law firms and legal professionals  
**Timeline**: MVP completion  
**Project Type**: Refactoring and enhancement of existing codebase  

---

## Executive Summary

**Refactor and enhance** the existing Vite-based React application into a **Next.js production-ready application** with Harvey.ai's clean, fresh, and sleek UI design language. The application serves legal professionals with two primary interfaces: `/upload` for document management and `/chat` for intelligent document querying. 

**Key Focus**: Transform the current functional prototype into a **production-grade, Harvey.ai-inspired interface** that matches the clean, professional aesthetic law firms expect from enterprise SaaS tools.

---

## Current Implementation Status

### ✅ **Existing Codebase (Vite + React)**
- **Landing Page**: Professional Harvey.ai-inspired design with hero section, features showcase
- **Document Management**: Upload, processing, and library functionality  
- **Chat Interface**: Message bubbles, conversation history, citation system
- **TypeScript Architecture**: Complete type system for documents, messages, citations
- **Custom Hooks**: `useDocuments`, `useChat`, `useDocumentChunking`
- **Professional Design Foundation**: Dark theme with orange/red accents, clean typography
- **Mock Backend Integration**: Ready for real API integration
- **Component Library**: Reusable UI components (DocumentUpload, ChatInterface, MessageBubble, etc.)

### 🔄 **Migration & Refactoring Required**
- **Vite → Next.js Migration**: Convert existing Vite application to Next.js framework
- **Harvey.ai UI Refinement**: Elevate current design to production-grade, Harvey.ai-level polish
- **Clean & Fresh Interface**: Implement sleek, modern UI that matches Harvey.ai's aesthetic standards
- **Production-Ready Components**: Refactor existing components for Next.js and enhanced UX
- **Page Routing**: Implement Next.js App Router for `/upload` and `/chat` pages

### 🟡 **Still Needs Implementation** 
- **Real Backend Integration**: Replace mock APIs with actual endpoints
- **Enhanced Error Handling**: Comprehensive error states and user feedback
- **Citation Preview Enhancement**: Document snippet viewing functionality
- **Performance Optimization**: Next.js-specific optimizations and production readiness

---

## Core Requirements

### 1. **Framework Migration & UI Overhaul**

#### 1.1 Next.js Migration Requirements
- **Framework Transition**: Convert existing Vite + React application to Next.js 14+ with App Router
- **Component Migration**: Refactor all existing React components to work seamlessly with Next.js
- **Build System**: Replace Vite build configuration with Next.js optimized build system
- **Routing Migration**: Convert current component-based routing to Next.js file-based routing system
- **Performance Optimization**: Leverage Next.js built-in optimizations (Image optimization, automatic code splitting, etc.)
- **TypeScript Integration**: Ensure all existing TypeScript configurations work with Next.js
- **Deployment Ready**: Configure for production deployment with Vercel or similar platforms

#### 1.2 Harvey.ai UI Standards Implementation
- **Visual Parity**: Match Harvey.ai's clean, fresh, and sleek visual design language
- **Production-Grade Polish**: Elevate current prototype to enterprise-SaaS quality
- **Clean Interface Design**: 
  - Crisp, minimal layouts with strategic use of whitespace
  - Professional color palette with subtle, sophisticated accents
  - Consistent typography hierarchy using modern, readable fonts
  - Subtle animations and micro-interactions for enhanced UX
- **Sleek Component Library**: 
  - Button styles with hover states matching Harvey.ai's aesthetic
  - Form inputs with clean borders and focus states
  - Cards and panels with subtle shadows and rounded corners
  - Professional loading states and progress indicators
- **Fresh, Modern Aesthetic**:
  - Light/dark mode toggle with Harvey.ai-inspired color schemes
  - Clean iconography and visual elements
  - Sophisticated data visualization for document status and chat history
  - Professional empty states and error handling interfaces

#### 1.3 Landing Page Enhancement
- **Hero Section**: Compelling, Harvey.ai-style hero with clear value proposition
- **Feature Showcase**: Clean, modern presentation of key features
- **Professional Branding**: Consistent with Harvey.ai's sophisticated brand approach
- **Call-to-Action**: Clear navigation to `/upload` and `/chat` functionality
- **Responsive Design**: Flawless experience across all device sizes
- **Performance**: Fast loading times with Next.js optimizations

### 2. **Application Routing Structure**

#### 2.1 Next.js App Router Architecture
```
app/
├── page.tsx                    → Landing Page (/)
├── upload/
│   └── page.tsx               → Document Upload & Management (/upload)
├── chat/
│   └── page.tsx               → Chat Interface (/chat)
├── layout.tsx                 → Root layout with navigation
└── globals.css                → Global styles
```

#### 2.2 Navigation Requirements
- **Header Navigation**: Consistent across `/upload` and `/chat` pages
- **Seamless Transitions**: Users should easily move between upload and chat
- **State Persistence**: Document state should persist across page navigation
- **Deep Linking**: Direct access to `/upload` and `/chat` routes

### 3. **Upload Page (/upload)**

#### 3.1 Core Functionality
- **Document Upload Interface**: 
  - Leverage existing `DocumentUpload.tsx` component
  - Drag-and-drop file selection
  - File picker browser fallback
  - Real-time upload progress indicators

#### 3.2 Document Management Interface
- **Document List/Table View**:
  - Document name, upload date, file size
  - Processing status: "Processing" | "Ready" | "Error"
  - Visual status indicators (loading spinners, checkmarks, error icons)
  - Document actions: View, Delete, Re-process (if failed)

#### 3.3 Backend Integration
- **Upload API**: `POST /documents/upload` (EXCETRA-36)
- **Document List API**: `GET /documents` for user's documents
- **Delete API**: `DELETE /documents/{id}`
- **Status Polling**: Real-time updates for processing status

#### 3.4 User Experience
- **Empty State**: Clear guidance when no documents uploaded
- **Progress Feedback**: Visual progress for multi-file uploads
- **Error Handling**: Clear messages for file size, format, or upload failures
- **Success States**: Confirmation when documents are ready for chat

### 4. **Chat Page (/chat)**

#### 4.1 Core Chat Interface
- **Conversation History**: 
  - Utilize existing `MessageBubble.tsx` components
  - User and assistant message distinction
  - Persistent conversation scrolling
  - Message timestamps

#### 4.2 Query Input & Submission
- **Text Input Area**: 
  - Multi-line text input for complex legal questions
  - Send button with keyboard shortcuts (Enter to send)
  - Character/token limits if applicable
  - Auto-resizing input field

#### 4.3 Backend Integration
- **Chat API**: `POST /chat/query` (EXCETRA-31, 32)
- **Request Payload**: User question + optional document scope
- **Response Handling**: Streaming or standard response with citations
- **Error Recovery**: Retry mechanisms for failed queries

#### 4.4 Loading & Feedback States
- **Query Processing**: "Thinking..." indicator or loading spinner
- **Response Streaming**: Progressive message building (if streaming supported)
- **Error States**: Clear error messages for API failures or no results

#### 4.5 Document Context Requirements
- **Prerequisites Check**: Verify user has uploaded documents before allowing queries
- **Empty State**: Prompt to upload documents if none available
- **Document Scope**: Option to query specific documents vs. all documents

### 5. **Citation System Enhancement**

#### 5.1 Citation Detection & Rendering
- **Pattern Recognition**: Detect citation formats like `[Document.pdf p. 3]`
- **Clickable Citations**: Transform citations into interactive elements
- **Visual Styling**: Distinct styling for citations within message text

#### 5.2 Citation Preview Functionality
- **Source Snippet Display**: Show relevant text excerpt on citation click
- **Document Context**: Display document name, page number, section
- **Preview Component**: Utilize existing `CitationViewer.tsx` or enhance
- **Preview Trigger**: Click, hover, or tap interactions

#### 5.3 Backend Integration
- **Snippet API**: `GET /documents/{id}/snippet` for citation content
- **Citation Metadata**: Include snippet data in chat response payload
- **Fallback Handling**: Graceful degradation if snippet unavailable

### 6. **Professional Design & UX**

#### 6.1 Harvey.ai Design Language Implementation
- **Typography**: Clean, readable sans-serif fonts matching Harvey.ai
- **Color Palette**: Professional blues, grays, whites with subtle orange accents
- **Layout**: Generous whitespace, clear information hierarchy
- **Interactive Elements**: Subtle hover states, professional button styling

#### 6.2 Responsive Design Requirements
- **Primary Target**: Desktop and tablet interfaces
- **Mobile Compatibility**: Functional on mobile (lower priority but not broken)
- **Breakpoint Strategy**: Tailwind CSS responsive utilities
- **Touch Interfaces**: Appropriate touch targets for tablet users

#### 6.3 Accessibility Standards
- **WCAG 2.1 AA Compliance**: Color contrast, keyboard navigation
- **Screen Reader Support**: Semantic HTML, ARIA attributes
- **Focus Management**: Clear focus indicators and logical tab order

### 7. **Error Handling & Edge Cases**

#### 7.1 Document Upload Errors
- **File Size Limits**: Clear messaging for oversized files
- **Unsupported Formats**: Guidance on supported file types (PDF, DOCX, XLSX)
- **Network Failures**: Retry mechanisms and clear error states
- **Processing Failures**: Document parsing or ingestion errors

#### 7.2 Chat Query Errors
- **No Documents**: Prompt to upload documents before querying
- **No Results**: "No information found in your documents" messaging
- **API Failures**: Network errors, rate limits, service unavailable
- **Malformed Queries**: Guidance for effective questioning

#### 7.3 Network & Performance
- **Offline Detection**: Indicate when application is offline
- **Slow Connections**: Progressive loading and timeout handling
- **Large Document Handling**: Progress indicators for processing

---

## Technical Implementation

### 8.1 **Next.js Routing Implementation**
```typescript
// Next.js App Router structure
app/
├── layout.tsx                 // Root layout with navigation
├── page.tsx                   // Landing page (/)
├── upload/
│   └── page.tsx              // Upload page (/upload)
└── chat/
    └── page.tsx              // Chat page (/chat)

// layout.tsx - Root layout
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <Navigation />
        {children}
      </body>
    </html>
  )
}
```

### 8.2 **State Management**
- **Global State**: Context API for documents and chat state
- **Persistence**: LocalStorage for conversation history
- **Sync Strategy**: Optimistic updates with backend sync

### 8.3 **API Integration Points**
```typescript
// Required API endpoints
POST /documents/upload     // File upload (EXCETRA-36)
GET  /documents           // List user documents
DELETE /documents/{id}    // Delete document
POST /chat/query          // Submit question (EXCETRA-31, 32)
GET  /documents/{id}/snippet // Citation content
```

### 8.4 **Next.js Component Architecture**
```
src/
├── pages/
│   ├── LandingPage.tsx
│   ├── UploadPage.tsx
│   └── ChatPage.tsx
├── components/
│   ├── Layout/
│   │   ├── Header.tsx
│   │   └── Navigation.tsx
│   ├── Upload/
│   │   ├── DocumentUpload.tsx (existing)
│   │   ├── DocumentLibrary.tsx (existing)
│   │   └── DocumentTable.tsx (new)
│   ├── Chat/
│   │   ├── ChatInterface.tsx (existing)
│   │   ├── MessageBubble.tsx (existing)
│   │   ├── CitationTag.tsx (existing)
│   │   └── CitationViewer.tsx (enhance)
│   └── Common/
│       ├── ErrorBoundary.tsx
│       ├── LoadingSpinner.tsx
│       └── EmptyState.tsx
```

---

## Success Metrics

### 9.1 **User Experience Metrics**
- Document upload success rate > 95%
- Average time from upload to "Ready" status visibility
- Chat query response time < 3 seconds
- Citation click-through rate

### 9.2 **Technical Performance**
- Page load times < 2 seconds
- File upload progress accuracy
- Mobile responsiveness score
- Accessibility compliance verification

### 9.3 **Error Handling Coverage**
- Graceful degradation for all identified error scenarios
- Clear user feedback for all error states
- Recovery paths for all failed operations

---

## Implementation Phases

### **Phase 1: Vite → Next.js Migration** (Week 1)
- [ ] Initialize Next.js 14+ project with App Router
- [ ] Migrate existing TypeScript configurations
- [ ] Convert Vite components to Next.js App Router structure
- [ ] Set up Next.js routing (`/`, `/upload`, `/chat` pages)
- [ ] Migrate Tailwind CSS configuration to Next.js
- [ ] Test basic functionality in Next.js environment

### **Phase 2: Harvey.ai UI Enhancement** (Week 2)
- [ ] Implement Harvey.ai-inspired design system
- [ ] Refactor existing components with production-grade polish
- [ ] Enhance landing page with sleek, modern aesthetic
- [ ] Add subtle animations and micro-interactions
- [ ] Implement light/dark mode toggle
- [ ] Responsive design optimization

### **Phase 3: Backend Integration & Core Features** (Week 3)
- [ ] Replace mock APIs with real backend endpoints
- [ ] Implement error handling for API failures
- [ ] Add real-time status updates for document processing
- [ ] Test file upload with backend API (EXCETRA-36)
- [ ] Implement citation click handlers and preview functionality
- [ ] Integrate document snippet API

### **Phase 4: Production Polish & Testing** (Week 4)
- [ ] Comprehensive error state testing
- [ ] Next.js performance optimization
- [ ] Accessibility audit and WCAG 2.1 AA compliance
- [ ] Cross-browser and device testing
- [ ] User acceptance testing
- [ ] Production deployment preparation

---

## Dependencies & Blockers

### **Backend Dependencies**
- **EXCETRA-36**: File upload API (In Review) - *Required for real upload functionality*
- **EXCETRA-31**: Retrieval query endpoint (Todo) - *Required for chat functionality*
- **EXCETRA-32**: LLM integration (Todo) - *Required for AI responses*

### **Optional Enhancements**
- **EXCETRA-43**: Auth middleware - *For user authentication*
- **EXCETRA-35**: Conversation context - *For follow-up questions*

---

## Quality Assurance

### **Testing Strategy**
- **Unit Tests**: Component testing with Jest/React Testing Library
- **Integration Tests**: API integration testing
- **E2E Tests**: Critical user flows (upload → chat workflow)
- **Accessibility Testing**: Screen reader and keyboard navigation
- **Cross-browser Testing**: Chrome, Firefox, Safari, Edge

### **Acceptance Criteria**
- [ ] Users can upload documents via `/upload` page
- [ ] Document processing status updates in real-time
- [ ] Users can ask questions via `/chat` page
- [ ] Citations are clickable and show source snippets
- [ ] Error states provide clear guidance and recovery paths
- [ ] Application works on desktop, tablet, and mobile (functional)
- [ ] Design matches Harvey.ai professional aesthetic
- [ ] All interactions follow WCAG 2.1 AA accessibility standards

---

## Risk Mitigation

### **Technical Risks**
- **Backend API Delays**: Maintain mock fallbacks during development
- **File Processing Issues**: Implement robust error handling and retry logic
- **Performance with Large Files**: Progressive loading and chunked uploads

### **UX Risks**
- **Complex Legal Queries**: Provide query guidance and examples
- **Citation Complexity**: Start with simple tooltip implementation
- **Mobile Experience**: Ensure core functionality works on small screens

---

## Future Enhancements (Post-MVP)

- **Advanced Document Management**: Folders, tags, bulk operations
- **Conversation Export**: PDF/Word export of chat sessions
- **Document Preview**: Full document viewer with highlight
- **Collaborative Features**: Shared document libraries
- **Advanced Search**: Full-text search across documents
- **Template Queries**: Pre-built legal question templates 