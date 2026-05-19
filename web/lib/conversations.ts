// Conversation persistence in localStorage. Chat transcripts stay on the
// user's device — they never leave the browser. The /privacy page covers the
// disclosure; the composer tooltip surfaces it inline.

import type { EvalNeighbour } from "@/lib/types";
import { isBrowser } from "@/lib/utils";

export interface ChatMeta {
  request_id: string;
  latency_ms: number;
  tokens_in: number;
  tokens_out: number;
  cache_hit: boolean;
  model: string;
  turn_count: number;
  eval_neighbours?: EvalNeighbour[];
  snapshot_served?: boolean;
}

export interface LeadContextRef {
  id: string;
  name: string;
  title: string;
  companyName: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  meta?: ChatMeta;
  exampleId?: string | null;
  stopped?: boolean;
  leadContext?: LeadContextRef;
}

export interface StoredConversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

const STORAGE_KEY = "lead-enrichment.conversations.v1";

export function loadConversations(): StoredConversation[] {
  if (!isBrowser()) return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isStoredConversation);
  } catch {
    return [];
  }
}

export function saveConversations(conversations: StoredConversation[]): void {
  if (!isBrowser()) return;
  let working = [...conversations];
  while (working.length > 0) {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(working));
      return;
    } catch (err) {
      // Quota exceeded: drop oldest and retry.
      working.sort((a, b) => a.updatedAt - b.updatedAt);
      working.shift();
      if (working.length === 0) {
        console.warn("conversations: localStorage quota exceeded; cleared", err);
        try {
          window.localStorage.removeItem(STORAGE_KEY);
        } catch {
          // give up
        }
        return;
      }
    }
  }
}

export function upsertConversation(
  conversations: StoredConversation[],
  next: StoredConversation,
): StoredConversation[] {
  const idx = conversations.findIndex((c) => c.id === next.id);
  if (idx === -1) return [next, ...conversations];
  const copy = [...conversations];
  copy[idx] = next;
  return copy;
}

export function deleteConversation(
  conversations: StoredConversation[],
  id: string,
): StoredConversation[] {
  return conversations.filter((c) => c.id !== id);
}

export function newId(): string {
  if (isBrowser() && typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

const TITLE_MAX_CHARS = 36;

export function autoTitle(
  messages: ChatMessage[],
  starterLabel?: string,
): string {
  if (starterLabel && starterLabel.trim()) return starterLabel.trim();
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser) return "Untitled conversation";
  const flat = firstUser.content.replace(/\s+/g, " ").trim();
  if (!flat) return "Untitled conversation";
  if (flat.length <= TITLE_MAX_CHARS) return flat;
  const truncated = flat.slice(0, TITLE_MAX_CHARS);
  const lastSpace = truncated.lastIndexOf(" ");
  const cut = lastSpace > TITLE_MAX_CHARS * 0.5 ? truncated.slice(0, lastSpace) : truncated;
  return `${cut.trimEnd()}…`;
}

export function sortByUpdated(
  conversations: StoredConversation[],
): StoredConversation[] {
  return [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
}

function isStoredConversation(value: unknown): value is StoredConversation {
  if (!value || typeof value !== "object") return false;
  const c = value as Record<string, unknown>;
  return (
    typeof c.id === "string" &&
    typeof c.title === "string" &&
    typeof c.createdAt === "number" &&
    typeof c.updatedAt === "number" &&
    Array.isArray(c.messages)
  );
}
