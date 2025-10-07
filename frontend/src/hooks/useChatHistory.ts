import { useState, useEffect, useCallback } from 'react';
import { ChatSession, Message } from '../types';

const CHAT_HISTORY_KEY = 'chat_history';
const MAX_SESSIONS = 20; // Limit number of stored sessions

export function useChatHistory() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load chat history from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(CHAT_HISTORY_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        // Convert date strings back to Date objects
        const sessionsWithDates = parsed.map((session: any) => ({
          ...session,
          createdAt: new Date(session.createdAt),
          updatedAt: new Date(session.updatedAt),
          messages: session.messages.map((message: any) => ({
            ...message,
            timestamp: new Date(message.timestamp)
          }))
        }));
        setSessions(sessionsWithDates);

        // Set current session to the most recent one
        if (sessionsWithDates.length > 0) {
          setCurrentSessionId(sessionsWithDates[0].id);
        }
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
    } finally {
      setIsLoaded(true);
    }
  }, []);

  // Save to localStorage whenever sessions change (but only after initial load)
  useEffect(() => {
    if (!isLoaded) return; // Don't save until we've loaded from localStorage

    try {
      localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(sessions));
    } catch (error) {
      console.error('Failed to save chat history:', error);
    }
  }, [sessions, isLoaded]);

  const generateSessionTitle = (firstMessage: string): string => {
    // Generate a title from the first user message (truncated)
    const title = firstMessage.trim().substring(0, 50);
    return title.length > 50 ? title + '...' : title;
  };

  const createNewSession = useCallback((): string => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };

    setSessions(prev => {
      // Add new session at the beginning and limit total sessions
      const updated = [newSession, ...prev].slice(0, MAX_SESSIONS);
      return updated;
    });

    setCurrentSessionId(newSession.id);
    return newSession.id;
  }, []);

  const addMessageToSession = useCallback((sessionId: string, message: Message) => {
    setSessions(prev => prev.map(session => {
      if (session.id === sessionId) {
        const updatedMessages = [...session.messages, message];

        // Update title if this is the first user message
        let updatedTitle = session.title;
        if (session.title === 'New Chat' && message.type === 'user' && updatedMessages.length === 1) {
          updatedTitle = generateSessionTitle(message.content);
        }

        return {
          ...session,
          title: updatedTitle,
          messages: updatedMessages,
          updatedAt: new Date()
        };
      }
      return session;
    }));
  }, []);

  const deleteSession = useCallback((sessionId: string) => {
    setSessions(prev => {
      const filtered = prev.filter(session => session.id !== sessionId);

      // If we deleted the current session, switch to the next available one
      if (sessionId === currentSessionId) {
        setCurrentSessionId(filtered.length > 0 ? filtered[0].id : null);
      }

      return filtered;
    });
  }, [currentSessionId]);

  const switchToSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId);
  }, []);

  const clearAllHistory = useCallback(() => {
    setSessions([]);
    setCurrentSessionId(null);
    localStorage.removeItem(CHAT_HISTORY_KEY);
  }, []);

  const getCurrentSession = useCallback((): ChatSession | null => {
    if (!currentSessionId) return null;
    return sessions.find(session => session.id === currentSessionId) || null;
  }, [sessions, currentSessionId]);

  const getCurrentMessages = useCallback((): Message[] => {
    const session = getCurrentSession();
    return session ? session.messages : [];
  }, [getCurrentSession]);

  // Auto-create first session if none exist (only after loading from localStorage)
  useEffect(() => {
    if (isLoaded && sessions.length === 0 && !currentSessionId) {
      createNewSession();
    }
  }, [isLoaded, sessions.length, currentSessionId, createNewSession]);

  return {
    sessions,
    currentSessionId,
    getCurrentSession,
    getCurrentMessages,
    createNewSession,
    addMessageToSession,
    deleteSession,
    switchToSession,
    clearAllHistory
  };
}