import React from 'react';
import { User, Bot, Copy, RotateCcw } from 'lucide-react';
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
        </div>

        <div className={`flex-1 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
          <div
            className={`inline-block p-4 rounded-2xl max-w-full shadow-lg ${
              message.type === 'user'
                ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white'
                : 'bg-stone-800 text-stone-100 border border-orange-600/20'
            }`}
          >
            <div className="prose prose-sm max-w-none">
              <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
            </div>

            {/* Citations section removed */}
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
