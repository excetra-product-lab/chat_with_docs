import React, { useState } from 'react';
import Image from 'next/image';
import { User, Bot, Copy, RotateCcw, FileText, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Message } from '../types';
// Citation imports removed

interface MessageBubbleProps {
  message: Message;
  onCopy?: () => void;
  onRegenerate?: () => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onCopy,
  onRegenerate
}) => {
  const [citationsExpanded, setCitationsExpanded] = useState(false);
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    onCopy?.();
  };

  return (
    <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`flex max-w-4xl ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
        <div className={`flex-shrink-0 ${message.type === 'user' ? 'ml-3' : 'mr-3'}`}>
          {message.type === 'user' && message.userImageUrl ? (
            <Image
              src={message.userImageUrl}
              alt="User profile"
              width={32}
              height={32}
              className="w-8 h-8 rounded-full shadow-lg object-cover ring-2 ring-blue-500/30"
            />
          ) : (
            <div className={`w-8 h-8 rounded-full flex items-center justify-center shadow-lg ${
              message.type === 'user'
                ? 'bg-gradient-to-br from-blue-600 to-blue-700'
                : 'bg-gradient-to-br from-orange-600 to-red-700'
            }`}>
              {message.type === 'user' ? (
                <User className="w-4 h-4 text-white" />
              ) : (
                <Bot className="w-4 h-4 text-white" />
              )}
            </div>
          )}
        </div>

        <div className={`flex-1 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
          <div
            className={`inline-block p-4 rounded-2xl max-w-full shadow-lg ${
              message.type === 'user'
                ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white'
                : 'bg-stone-800 text-stone-100 border border-orange-600/20'
            }`}
          >
            <div className="prose prose-sm max-w-none prose-invert">
              {message.type === 'assistant' ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>,
                    ul: ({ children }) => <ul className="mb-3 ml-4 list-disc space-y-1">{children}</ul>,
                    ol: ({ children }) => <ol className="mb-3 ml-4 list-decimal space-y-1">{children}</ol>,
                    li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                    strong: ({ children }) => <strong className="font-semibold text-orange-300">{children}</strong>,
                    em: ({ children }) => <em className="italic text-stone-200">{children}</em>,
                    code: ({ children }) => <code className="px-1.5 py-0.5 bg-stone-900/60 text-orange-300 rounded text-sm font-mono border border-stone-700/50">{children}</code>,
                    pre: ({ children }) => <pre className="bg-stone-900/60 p-3 rounded-lg overflow-x-auto border border-stone-700/50 mb-3">{children}</pre>,
                    blockquote: ({ children }) => <blockquote className="border-l-3 border-orange-600/50 pl-4 my-3 italic text-stone-300">{children}</blockquote>,
                    h1: ({ children }) => <h1 className="text-xl font-bold text-orange-300 mb-3 mt-4 first:mt-0">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-lg font-semibold text-orange-300 mb-2 mt-3 first:mt-0">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-base font-semibold text-orange-300 mb-2 mt-3 first:mt-0">{children}</h3>,
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              ) : (
                <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
              )}
            </div>

            {/* Confidence Score Display */}
            {message.type === 'assistant' && message.confidence !== undefined && (
              <div className="mt-3 pt-3 border-t border-orange-600/20">
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-stone-400">Confidence:</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 h-1.5 bg-stone-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-300 ${
                          message.confidence >= 0.8 ? 'bg-green-500' :
                          message.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${(message.confidence * 100)}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-stone-300 font-medium">
                      {Math.round(message.confidence * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Citations Display */}
            {message.type === 'assistant' && message.citations && message.citations.length > 0 && (
              <div className="mt-3 pt-3 border-t border-orange-600/20">
                <div className="space-y-2">
                  <button
                    onClick={() => setCitationsExpanded(!citationsExpanded)}
                    className="flex items-center space-x-2 text-xs text-stone-400 hover:text-stone-300 transition-colors"
                  >
                    <FileText className="w-3 h-3" />
                    <span>Sources ({message.citations.length})</span>
                    {citationsExpanded ? (
                      <ChevronUp className="w-3 h-3" />
                    ) : (
                      <ChevronDown className="w-3 h-3" />
                    )}
                  </button>

                  {citationsExpanded && (
                    <div className="space-y-2">
                      {message.citations.map((citation, index) => (
                        <div 
                          key={index}
                          className="p-2 bg-stone-900/50 rounded-lg border border-stone-700/50"
                        >
                          <div className="flex items-center space-x-2 mb-1">
                            <FileText className="w-3 h-3 text-orange-400" />
                            <span className="text-xs font-medium text-stone-300">
                              {citation.document_name}
                            </span>
                            {citation.page_number && (
                              <span className="text-xs text-stone-500">
                                Page {citation.page_number}
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-stone-400 leading-relaxed">
                            "{citation.chunk_text}"
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className={`flex items-center mt-2 space-x-3 ${
            message.type === 'user' ? 'justify-end' : 'justify-start'
          }`}>
            <span className="text-xs text-stone-500">{formatTime(message.timestamp)}</span>

            {message.type === 'assistant' && (
              <div className="flex space-x-1">
                <button
                  onClick={handleCopy}
                  className="p-1 text-stone-500 hover:text-stone-300 transition-colors"
                  title="Copy message"
                >
                  <Copy className="w-3 h-3" />
                </button>
                <button
                  onClick={onRegenerate}
                  className="p-1 text-stone-500 hover:text-stone-300 transition-colors"
                  title="Regenerate response"
                >
                  <RotateCcw className="w-3 h-3" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
