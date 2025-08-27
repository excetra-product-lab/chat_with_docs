import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { Message } from '../types';
import { MessageBubble } from './MessageBubble';
import { MessageSkeleton, LoadingSpinner } from './Loading';

interface ChatInterfaceProps {
  messages: Message[];
  isLoading: boolean;
  onSendMessage: (message: string) => void;
  hasDocuments: boolean;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  isLoading,
  onSendMessage,
  hasDocuments
}) => {
  const [inputMessage, setInputMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputMessage.trim() && !isLoading) {
      onSendMessage(inputMessage.trim());
      setInputMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const suggestedQuestions = [
    "What are the key terms of the employment contract?",
    "What are the tenant's obligations in the lease?",
    "What confidentiality requirements are specified in the NDA?",
    "What is the termination clause in the contract?"
  ];

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-midnight-950 to-slate-900">
      <div className="px-6 py-6 border-b border-slate-800/50 bg-gradient-to-r from-midnight-950/80 to-slate-900/80 backdrop-blur-sm">
        <h1 className="text-2xl font-bold text-slate-100 mb-1 tracking-tight">Excetera Chat Bot</h1>
        <p className="text-slate-400 font-light">Ask questions about your uploaded documents</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            {hasDocuments ? (
              <>
                <div className="w-16 h-16 bg-gradient-to-br from-violet-600 to-electric-600 rounded-full flex items-center justify-center mb-4 shadow-glow">
                  <span className="text-2xl">⚖️</span>
                </div>
                <h3 className="text-xl font-semibold text-slate-100 mb-2">Ready to assist</h3>
                <p className="text-slate-400 mb-6 max-w-md font-light">
                  Your documents are processed and ready. Ask me anything about their contents.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
                  {suggestedQuestions.map((question, index) => (
                    <button
                      key={index}
                      onClick={() => setInputMessage(question)}
                      className="p-3 text-left glass-effect text-slate-300 rounded-xl hover:border-violet-500/50 transition-all duration-300 hover:transform hover:scale-105 font-light"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <>
                <div className="w-16 h-16 bg-slate-800/50 rounded-full flex items-center justify-center mb-4 border border-slate-700/50">
                  <span className="text-2xl">📄</span>
                </div>
                <h3 className="text-xl font-semibold text-slate-100 mb-2">Upload documents to get started</h3>
                <p className="text-slate-400 max-w-md font-light">
                  Upload your legal documents to the left panel and I'll help you analyze and understand their contents.
                </p>
              </>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-violet-600 to-electric-600 rounded-full flex items-center justify-center shadow-glow">
                    <LoadingSpinner size="small" color="white" />
                  </div>
                  <div className="glass-effect rounded-2xl p-4 border-violet-500/20">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-electric-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-6 border-t border-slate-800/50 bg-gradient-to-r from-midnight-950/80 to-slate-900/80 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="flex space-x-4">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={hasDocuments ? "Ask a question about your documents..." : "Upload documents first to start chatting"}
              className="w-full px-4 py-3 glass-effect text-slate-100 rounded-xl focus:border-violet-500 focus:ring-1 focus:ring-violet-500/50 resize-none transition-all duration-300 placeholder-slate-400 font-light"
              rows={1}
              disabled={!hasDocuments || isLoading}
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
          </div>
          <button
            type="submit"
            disabled={!inputMessage.trim() || !hasDocuments || isLoading}
            className="px-6 py-3 bg-gradient-to-r from-violet-600 to-electric-600 hover:from-violet-500 hover:to-electric-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500 text-white rounded-xl transition-all duration-300 flex items-center justify-center shadow-lg hover:shadow-glow transform hover:scale-105 disabled:transform-none disabled:hover:scale-100"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
