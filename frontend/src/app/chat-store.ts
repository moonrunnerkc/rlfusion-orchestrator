// Author: Bradley R. Kinnard
// LocalStorage-backed chat session store. Pure data, no React.

import type { ChatSession, Message } from '../types/contracts';

const STORAGE_KEY = 'rlfusion_chats';

export function generateTitle(messages: Message[]): string {
  const firstUserMsg = messages.find((m) => m.role === 'user');
  if (!firstUserMsg) return 'New Chat';
  const text = firstUserMsg.text.slice(0, 40);
  return text.length < firstUserMsg.text.length ? `${text}...` : text;
}

export function loadChats(): ChatSession[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? (JSON.parse(stored) as ChatSession[]) : [];
  } catch {
    return [];
  }
}

export function saveChats(chats: ChatSession[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

export function createSession(title = 'New Chat'): ChatSession {
  const now = Date.now();
  return { id: now.toString(), title, messages: [], createdAt: now, updatedAt: now };
}
