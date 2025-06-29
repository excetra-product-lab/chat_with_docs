import React, { useState } from 'react';
import { ExternalLink } from 'lucide-react';
import { Citation } from '../types';

interface CitationTagProps {
  citation: Citation;
}

export const CitationTag: React.FC<CitationTagProps> = ({ citation }) => {
  const [showPreview, setShowPreview] = useState(false);

  const citationText = citation.page 
    ? `${citation.documentName}, p.${citation.page}`
    : citation.section
    ? `${citation.documentName}, ${citation.section}`
    : citation.documentName;

  const previewText = citation.section === 'Section 8.2'
    ? '"Either party may terminate this employment agreement by providing written notice of thirty (30) days in advance to the other party. Such notice shall be in writing and delivered..."'
    : citation.section === 'Section 3.1'
    ? '"Tenant shall pay rent in the amount specified herein on or before the first (1st) day of each calendar month during the term of this lease..."'
    : '"The receiving party agrees to hold all confidential information in strict confidence and not to disclose such information to any third party..."';

  return (
    <div className="relative inline-block">
      <button
        className="inline-flex items-center px-2 py-1 bg-orange-600/20 hover:bg-orange-600/30 text-orange-300 text-xs rounded border border-orange-600/40 hover:border-orange-500 transition-all duration-200 shadow-sm"
        onMouseEnter={() => setShowPreview(true)}
        onMouseLeave={() => setShowPreview(false)}
        onClick={() => setShowPreview(!showPreview)}
      >
        <ExternalLink className="w-3 h-3 mr-1" />
        [{citationText}]
      </button>
      
      {showPreview && (
        <div className="absolute bottom-full left-0 mb-2 w-80 bg-stone-800 border border-stone-600 rounded-lg p-3 shadow-xl z-10">
          <div className="text-xs text-stone-300 mb-1 font-medium">
            {citation.documentName}
          </div>
          <div className="text-xs text-stone-400 mb-2">
            {citation.section || `Page ${citation.page}`}
          </div>
          <div className="text-sm text-stone-200 italic">
            {previewText}
          </div>
          <div className="absolute top-full left-4 -mt-1">
            <div className="border-4 border-transparent border-t-stone-800"></div>
          </div>
        </div>
      )}
    </div>
  );
};