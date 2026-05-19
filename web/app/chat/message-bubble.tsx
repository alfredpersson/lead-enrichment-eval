"use client";

import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { LowSimilarityBadge } from "@/app/_shared/low-similarity-badge";
import type { ChatMessage } from "@/lib/conversations";
import { formatNumber } from "@/lib/utils";
import styles from "./chat.module.css";

const ASSISTANT_NAME = "Lead Copilot";
const ASSISTANT_INITIAL = "L";
const COPIED_FLASH_MS = 1500;

interface Props {
  message: ChatMessage;
  showMetrics: boolean;
  editing: boolean;
  editingDraft: string;
  onEditDraftChange: (text: string) => void;
  onStartEdit: () => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
  isStreaming: boolean;
  onStop?: () => void;
  onRerunLive?: () => void;
}

export function MessageBubble({
  message,
  showMetrics,
  editing,
  editingDraft,
  onEditDraftChange,
  onStartEdit,
  onSaveEdit,
  onCancelEdit,
  isStreaming,
  onStop,
  onRerunLive,
}: Props) {
  const isUser = message.role === "user";
  const finalAssistant = !isUser && (message.meta || message.stopped);
  const showStreamingHint =
    !isUser && !message.meta && !message.stopped && message.content.length === 0;
  const [copied, setCopied] = useState(false);
  const copyTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const editRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    return () => {
      if (copyTimeout.current) clearTimeout(copyTimeout.current);
    };
  }, []);

  useEffect(() => {
    if (editing && editRef.current) {
      editRef.current.focus();
      const len = editRef.current.value.length;
      editRef.current.setSelectionRange(len, len);
    }
  }, [editing]);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      if (copyTimeout.current) clearTimeout(copyTimeout.current);
      copyTimeout.current = setTimeout(() => setCopied(false), COPIED_FLASH_MS);
    } catch {
      // clipboard unavailable; leave button as is
    }
  }

  function handleEditKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      onSaveEdit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onCancelEdit();
    }
  }

  return (
    <article
      className={`${styles.bubble} ${
        isUser ? styles.bubbleUser : styles.bubbleAssistant
      }`}
    >
      <header className={styles.bubbleHeader}>
        {isUser ? (
          <span className={styles.bubbleHeaderIdent}>
            <span className={styles.bubbleHeaderText}>You</span>
            {message.leadContext && (
              <span className={styles.bubbleContextTag}>
                Re: {message.leadContext.name}
              </span>
            )}
          </span>
        ) : (
          <span className={styles.bubbleHeaderIdent}>
            <span className={styles.avatar} aria-hidden="true">
              {ASSISTANT_INITIAL}
            </span>
            <span className={styles.bubbleHeaderText}>{ASSISTANT_NAME}</span>
          </span>
        )}
        {message.stopped && (
          <span className={styles.stoppedBadge} aria-label="stopped">
            stopped
          </span>
        )}
      </header>

      {editing ? (
        <div className={styles.editor}>
          <textarea
            ref={editRef}
            value={editingDraft}
            onChange={(e) => onEditDraftChange(e.target.value)}
            onKeyDown={handleEditKey}
            rows={Math.min(12, Math.max(3, editingDraft.split("\n").length + 1))}
          />
          <div className={styles.editorActions}>
            <button
              type="button"
              className={styles.editorCancel}
              onClick={onCancelEdit}
            >
              Cancel
            </button>
            <button
              type="button"
              className={styles.editorSave}
              onClick={onSaveEdit}
              disabled={!editingDraft.trim()}
            >
              Save and resend
            </button>
          </div>
        </div>
      ) : (
        <div className={styles.bubbleBody}>
          {isUser ? (
            <span className={styles.userText}>{message.content}</span>
          ) : message.content ? (
            <div className={styles.markdown}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: ({ children, ...props }) => (
                    <a {...props} target="_blank" rel="noreferrer">
                      {children}
                    </a>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          ) : showStreamingHint ? (
            <span className={styles.streamingDot} aria-label="thinking">
              …
            </span>
          ) : null}
        </div>
      )}

      {!editing && (
        <footer className={styles.bubbleFoot}>
          {isUser && !isStreaming && (
            <button
              type="button"
              className={styles.bubbleAction}
              onClick={onStartEdit}
            >
              Edit
            </button>
          )}
          {finalAssistant && (
            <button
              type="button"
              className={styles.bubbleAction}
              onClick={handleCopy}
            >
              {copied ? "Copied" : "Copy"}
            </button>
          )}
          {finalAssistant && message.meta?.snapshot_served && (
            <span
              className={styles.snapshotBadge}
              title="Cached starter response served from a committed snapshot."
            >
              Cached starter
            </span>
          )}
          {onRerunLive && (
            <button
              type="button"
              className={styles.bubbleAction}
              onClick={onRerunLive}
              title="Bypass the cached snapshot and run this starter against the live API."
            >
              Re-run live
            </button>
          )}
          {!isUser && isStreaming && onStop && (
            <button
              type="button"
              className={styles.bubbleStop}
              onClick={onStop}
            >
              Stop
            </button>
          )}
          {showMetrics && message.meta && (
            <span className={styles.bubbleMeta}>
              <span>{formatNumber(message.meta.latency_ms)} ms</span>
              <span aria-hidden="true">·</span>
              <span>{formatNumber(message.meta.tokens_in)} in</span>
              <span aria-hidden="true">·</span>
              <span>{formatNumber(message.meta.tokens_out)} out</span>
              <span aria-hidden="true">·</span>
              <span>turn {message.meta.turn_count}</span>
              {message.meta.cache_hit && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>cache hit</span>
                </>
              )}
            </span>
          )}
        </footer>
      )}

      {showMetrics &&
        message.meta?.eval_neighbours &&
        message.meta.eval_neighbours.length > 0 && (
          <section
            className={styles.bubbleNeighbours}
            aria-label="Eval neighbours"
          >
            <span className={styles.bubbleNeighboursLabel}>
              Eval neighbours
            </span>
            <LowSimilarityBadge
              neighbours={message.meta.eval_neighbours}
              className={styles.bubbleLowSim}
            />
            <ul className={styles.bubbleNeighbourList}>
              {message.meta.eval_neighbours.map((n) => (
                <li key={n.id} className={styles.bubbleNeighbourRow}>
                  <span>#{n.id}</span>
                  <span>sim {n.similarity.toFixed(2)}</span>
                  <span>
                    fit {n.score === null ? "—" : n.score.toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}
    </article>
  );
}
