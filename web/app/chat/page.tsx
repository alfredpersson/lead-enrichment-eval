"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { EXEMPLARS, type Exemplar } from "@/lib/exemplars";
import styles from "./chat.module.css";

const INPUT_CAP = 4000;
const COUNTER_THRESHOLD = 3000;

// Chat-specific starter copy. The text pasted into the composer is the
// exemplar's profile+company verbatim, but the card label and teaser are
// reframed as a salesperson workflow rather than an eval category.
const STARTER_COPY: Record<string, { label: string; teaser: string }> = {
  "1": {
    label: "Qualify a VP Product",
    teaser: "Series B B2B SaaS, AI feature already shipped",
  },
  "2": {
    label: "Decide on a borderline founder",
    teaser: "Series A, consumer-led with a growing B2B side",
  },
  "3": {
    label: "Triage a sparse profile",
    teaser: "Freelance designer, two-line bio",
  },
  "4": {
    label: "Qualify an engineering leader",
    teaser: "Director of Engineering, owns an AI feature",
  },
  "5": {
    label: "Qualify a non-English profile",
    teaser: "Nordic SaaS, profile written in Swedish",
  },
};

interface ChatMeta {
  request_id: string;
  latency_ms: number;
  tokens_in: number;
  tokens_out: number;
  cache_hit: boolean;
  model: string;
  turn_count: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  meta?: ChatMeta;
  exampleId?: string | null;
}

interface ErrorState {
  code: string;
  message: string;
}

function exemplarToComposerText(ex: Exemplar): string {
  return ex.company ? `${ex.profile}\n\n${ex.company}` : ex.profile;
}

function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [composer, setComposer] = useState("");
  const [pendingExampleId, setPendingExampleId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<ErrorState | null>(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const totals = useMemo(() => {
    let turns = 0;
    let latency = 0;
    let tokensIn = 0;
    let tokensOut = 0;
    for (const m of messages) {
      if (m.role === "assistant" && m.meta) {
        turns += 1;
        latency += m.meta.latency_ms;
        tokensIn += m.meta.tokens_in;
        tokensOut += m.meta.tokens_out;
      }
    }
    return { turns, latency, tokensIn, tokensOut };
  }, [messages]);

  const send = useCallback(async () => {
    const text = composer.trim();
    if (!text || sending) return;
    if (text.length > INPUT_CAP) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError(null);
    const exampleId = messages.length === 0 ? pendingExampleId : null;
    const userMsg: ChatMessage = {
      role: "user",
      content: text,
      exampleId,
    };
    const conversation = [...messages, userMsg];
    setMessages([...conversation, { role: "assistant", content: "" }]);
    setComposer("");
    setPendingExampleId(null);
    setSending(true);

    let assistantText = "";
    let assistantMeta: ChatMeta | undefined;
    let errorEvent: ErrorState | null = null;
    let terminal = false;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          messages: conversation.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          example_id: exampleId,
        }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        setError({
          code: `http_${res.status}`,
          message: `Upstream returned ${res.status}.`,
        });
        setMessages((prev) => prev.slice(0, -1));
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep = buffer.indexOf("\n\n");
        while (sep !== -1) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          for (const line of frame.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const json = line.slice(5).trim();
            if (!json) continue;
            try {
              const event = JSON.parse(json) as
                | { type: "text"; delta: string }
                | { type: "done"; meta: ChatMeta }
                | { type: "error"; code: string; message: string };
              if (event.type === "text") {
                assistantText += event.delta;
                setMessages((prev) => {
                  const next = [...prev];
                  const last = next[next.length - 1];
                  if (last?.role === "assistant" && !last.meta) {
                    next[next.length - 1] = { ...last, content: assistantText };
                  }
                  return next;
                });
              } else if (event.type === "done") {
                assistantMeta = event.meta;
                terminal = true;
              } else if (event.type === "error") {
                errorEvent = { code: event.code, message: event.message };
                terminal = true;
              }
            } catch {
              // ignore malformed frame
            }
          }
          sep = buffer.indexOf("\n\n");
        }
      }

      if (errorEvent) {
        setError(errorEvent);
        setMessages((prev) => prev.slice(0, -1));
        return;
      }

      if (!terminal) {
        setError({
          code: "stream_ended",
          message: "Stream ended without a completion event.",
        });
        setMessages((prev) => prev.slice(0, -1));
        return;
      }

      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = {
            ...last,
            content: assistantText,
            meta: assistantMeta,
          };
        }
        return next;
      });
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setError({
        code: "network",
        message: (err as Error).message ?? "Network error",
      });
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setSending(false);
      abortRef.current = null;
    }
  }, [composer, messages, pendingExampleId, sending]);

  const useStarter = useCallback((ex: Exemplar) => {
    setComposer(exemplarToComposerText(ex));
    setPendingExampleId(ex.id);
    composerRef.current?.focus();
  }, []);

  const resetConversation = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setMessages([]);
    setComposer("");
    setPendingExampleId(null);
    setSending(false);
    setError(null);
  }, []);

  const onKey = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        void send();
      }
    },
    [send],
  );

  const trimmedLen = composer.trim().length;
  const overCap = composer.length > INPUT_CAP;
  const canSend = !sending && trimmedLen > 0 && !overCap;

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div className={styles.headerCopy}>
          <h1 className={styles.title}>Talk through a lead</h1>
          <p className={styles.lede}>
            Paste a profile or describe the lead. Ask follow-ups, qualify
            against the ICP, and walk away with a draft outreach hook.
          </p>
        </div>
        <button
          type="button"
          className={`${styles.metricsToggle} ${
            showMetrics ? styles.metricsToggleActive : ""
          }`}
          onClick={() => setShowMetrics((v) => !v)}
          aria-pressed={showMetrics}
        >
          {showMetrics ? "Hide metrics" : "Show metrics"}
        </button>
      </header>

      <section className={styles.thread} ref={threadRef} aria-live="polite">
        {messages.length === 0 ? (
          <div className={styles.emptyState}>
            <p className={styles.emptyHint}>
              Try one of these to get started, or paste your own profile below.
            </p>
            <div className={styles.starters}>
              {EXEMPLARS.map((ex) => {
                const copy = STARTER_COPY[ex.id] ?? {
                  label: ex.label,
                  teaser: ex.teaser,
                };
                return (
                  <button
                    type="button"
                    className={styles.starter}
                    onClick={() => useStarter(ex)}
                    key={ex.id}
                  >
                    <span className={styles.starterLabel}>{copy.label}</span>
                    <span className={styles.starterTeaser}>{copy.teaser}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <div className={styles.messages}>
            {messages.map((m, i) => (
              <MessageBubble key={i} message={m} showMetrics={showMetrics} />
            ))}
          </div>
        )}
      </section>

      {error && (
        <div className={styles.errorBanner} role="alert">
          <strong>{error.code}</strong> · {error.message}
        </div>
      )}

      <section className={styles.composerCard}>
        <textarea
          ref={composerRef}
          value={composer}
          onChange={(e) => {
            setComposer(e.target.value);
            if (pendingExampleId) setPendingExampleId(null);
          }}
          onKeyDown={onKey}
          placeholder="Paste a profile (and optionally a company description), or pick a starter above."
          rows={6}
          disabled={sending}
          aria-label="Message"
        />
        <div className={styles.composerRow}>
          {composer.length >= COUNTER_THRESHOLD && (
            <span
              className={`${styles.counter} ${
                overCap ? styles.counterOver : ""
              }`}
              aria-live="polite"
            >
              {formatNumber(composer.length)} / {formatNumber(INPUT_CAP)} chars
              {overCap && " — over cap"}
            </span>
          )}
          <span className={styles.privacy}>Inputs are not stored.</span>
          <div className={styles.composerActions}>
            {messages.length > 0 && (
              <button
                type="button"
                className={styles.resetButton}
                onClick={resetConversation}
                disabled={sending}
              >
                New conversation
              </button>
            )}
            <button
              type="button"
              className={styles.sendButton}
              onClick={() => void send()}
              disabled={!canSend}
            >
              {sending ? "Sending…" : "Send"}
            </button>
          </div>
        </div>
      </section>

      {showMetrics && <TotalsBar totals={totals} />}
    </main>
  );
}

function MessageBubble({
  message,
  showMetrics,
}: {
  message: ChatMessage;
  showMetrics: boolean;
}) {
  const isUser = message.role === "user";
  const isStreaming = !isUser && !message.meta && message.content.length === 0;
  return (
    <article
      className={`${styles.bubble} ${
        isUser ? styles.bubbleUser : styles.bubbleAssistant
      }`}
    >
      <header className={styles.bubbleHeader}>
        {isUser ? "You" : "Assistant"}
      </header>
      <div className={styles.bubbleBody}>
        {message.content ? (
          message.content
        ) : isStreaming ? (
          <span className={styles.streamingDot} aria-label="streaming">
            …
          </span>
        ) : (
          ""
        )}
        {!isUser && !message.meta && message.content && (
          <span className={styles.streamingCaret} aria-hidden="true">
            ▍
          </span>
        )}
      </div>
      {message.meta && showMetrics && (
        <footer className={styles.bubbleMeta}>
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
        </footer>
      )}
    </article>
  );
}

function TotalsBar({
  totals,
}: {
  totals: { turns: number; latency: number; tokensIn: number; tokensOut: number };
}) {
  if (totals.turns === 0) return null;
  return (
    <aside className={styles.totals} aria-label="Conversation totals">
      <span className={styles.totalsLabel}>Totals</span>
      <span>
        {totals.turns} turn{totals.turns === 1 ? "" : "s"}
      </span>
      <span aria-hidden="true">·</span>
      <span>{formatNumber(totals.latency)} ms</span>
      <span aria-hidden="true">·</span>
      <span>{formatNumber(totals.tokensIn)} tokens in</span>
      <span aria-hidden="true">·</span>
      <span>{formatNumber(totals.tokensOut)} tokens out</span>
    </aside>
  );
}
