import React, { useState } from 'react';
import { MessageSquare, Plus, X, MoreVertical, Trash2, Calendar } from 'lucide-react';
import { ChatSession } from '../types';

interface ChatHistoryProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  onCreateNewSession: () => void;
  onSwitchSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onClearAll: () => void;
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({
  sessions,
  currentSessionId,
  onCreateNewSession,
  onSwitchSession,
  onDeleteSession,
  onClearAll
}) => {
  const [showDropdown, setShowDropdown] = useState<string | null>(null);

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      });
    } else if (days === 1) {
      return 'Yesterday';
    } else if (days < 7) {
      return `${days} days ago`;
    } else {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
      });
    }
  };

  return (
    <div className="h-full bg-gradient-to-br from-slate-900 to-slate-800 border-r border-slate-700/50">
      {/* Header */}
      <div className="p-4 border-b border-slate-700/50">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-slate-100 flex items-center">
            <MessageSquare className="w-5 h-5 mr-2 text-violet-400" />
            Chat History
          </h2>
          <button
            onClick={onCreateNewSession}
            className="p-2 text-slate-400 hover:text-violet-400 hover:bg-slate-800/50 rounded-lg transition-all duration-200"
            title="New Chat"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        {sessions.length > 0 && (
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>{sessions.length} conversation{sessions.length !== 1 ? 's' : ''}</span>
            <button
              onClick={onClearAll}
              className="text-red-400 hover:text-red-300 transition-colors"
              title="Clear all history"
            >
              Clear all
            </button>
          </div>
        )}
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          <div className="p-4 text-center text-slate-500">
            <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No chat history yet</p>
            <p className="text-xs mt-1">Start a conversation to see it here</p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`group relative p-3 rounded-lg cursor-pointer transition-all duration-200 ${
                  session.id === currentSessionId
                    ? 'bg-gradient-to-r from-violet-600/20 to-electric-600/20 border border-violet-500/30'
                    : 'hover:bg-slate-800/50 border border-transparent'
                }`}
                onClick={() => onSwitchSession(session.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className={`text-sm font-medium truncate ${
                      session.id === currentSessionId
                        ? 'text-violet-200'
                        : 'text-slate-200 group-hover:text-slate-100'
                    }`}>
                      {session.title}
                    </h3>

                    <div className="flex items-center mt-1 text-xs text-slate-500">
                      <Calendar className="w-3 h-3 mr-1" />
                      {formatDate(session.updatedAt)}
                      <span className="mx-2">•</span>
                      <span>{session.messages.length} message{session.messages.length !== 1 ? 's' : ''}</span>
                    </div>

                    {/* Preview of last message */}
                    {session.messages.length > 0 && (
                      <p className="text-xs text-slate-400 mt-1 truncate">
                        {session.messages[session.messages.length - 1].content}
                      </p>
                    )}
                  </div>

                  {/* Actions dropdown */}
                  <div className="relative ml-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowDropdown(showDropdown === session.id ? null : session.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-slate-300 rounded transition-all duration-200"
                    >
                      <MoreVertical className="w-3 h-3" />
                    </button>

                    {showDropdown === session.id && (
                      <div className="absolute right-0 top-full mt-1 w-32 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-10">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(session.id);
                            setShowDropdown(null);
                          }}
                          className="w-full px-3 py-2 text-left text-xs text-red-400 hover:bg-slate-700/50 rounded-lg flex items-center"
                        >
                          <Trash2 className="w-3 h-3 mr-2" />
                          Delete
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Click outside to close dropdown */}
      {showDropdown && (
        <div
          className="fixed inset-0 z-5"
          onClick={() => setShowDropdown(null)}
        />
      )}
    </div>
  );
};