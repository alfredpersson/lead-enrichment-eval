"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { StoredConversation } from "@/lib/conversations";
import { sortByUpdated } from "@/lib/conversations";
import styles from "./chat.module.css";

interface Props {
  conversations: StoredConversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

function relativeTime(ts: number, now: number): string {
  const diffMs = Math.max(0, now - ts);
  const sec = Math.floor(diffMs / 1000);
  if (sec < 60) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day}d ago`;
  return new Date(ts).toLocaleDateString();
}

export function HistoryMenu({
  conversations,
  activeId,
  onSelect,
  onDelete,
}: Props) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const now = Date.now();
  const ordered = useMemo(() => sortByUpdated(conversations), [conversations]);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className={styles.historyMenu} ref={wrapRef}>
      <button
        type="button"
        className={styles.historyTrigger}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        disabled={ordered.length === 0}
      >
        History
        {ordered.length > 0 ? ` (${ordered.length})` : ""}
        <span aria-hidden="true">{open ? " ▴" : " ▾"}</span>
      </button>
      {open && ordered.length > 0 && (
        <div className={styles.historyDropdown} role="menu">
          <ol className={styles.historyList}>
            {ordered.map((c) => {
              const active = c.id === activeId;
              return (
                <li key={c.id} className={styles.historyItem}>
                  <button
                    type="button"
                    className={`${styles.historyButton} ${
                      active ? styles.historyButtonActive : ""
                    }`}
                    onClick={() => {
                      onSelect(c.id);
                      setOpen(false);
                    }}
                  >
                    <span className={styles.historyTitle}>{c.title}</span>
                    <span className={styles.historyTime}>
                      {relativeTime(c.updatedAt, now)}
                    </span>
                  </button>
                  <button
                    type="button"
                    className={styles.historyDelete}
                    onClick={() => onDelete(c.id)}
                    aria-label={`Delete ${c.title}`}
                    title="Delete conversation"
                  >
                    ×
                  </button>
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </div>
  );
}
