"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { EXEMPLARS, type Exemplar } from "@/lib/exemplars";
import type {
  Action,
  Claim,
  EnrichOutput,
  StreamEvent,
} from "@/lib/types";
import styles from "./integrated.module.css";

const PROFILE_CAP = 4000;
const COMPANY_CAP = 2000;

const ACTION_LABELS: Record<Action, string> = {
  auto_add: "Add to prospects",
  propose: "Propose to user",
  discard: "Discard",
  refuse: "Refuse",
};

const ACTION_CHIP_LABELS: Record<Action, string> = {
  auto_add: "Auto-add",
  propose: "Propose",
  discard: "Discard",
  refuse: "Refuse",
};

const DIMENSION_LABELS: Array<[keyof EnrichOutput["fit_score"]["dimensions"], string]> = [
  ["stage_match", "Stage"],
  ["headcount_match", "Headcount"],
  ["arr_match", "ARR"],
  ["product_shape_match", "Product shape"],
  ["role_match", "Role"],
];

type RowStatus = "idle" | "streaming" | "done" | "error";

type ErrorState = { code: string; message: string };

interface Row {
  id: string;
  source: "exemplar" | "paste";
  label: string;
  title: string;
  companyName: string;
  profile: string;
  company: string | null;
  status: RowStatus;
  selected: boolean;
  expanded: boolean;
  underTheHood: boolean;
  thinking: string;
  result: EnrichOutput | null;
  error: ErrorState | null;
  confirmedAt: number | null;
  hoveredQuote: string | null;
}

function firstLine(text: string): string {
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed) return trimmed;
  }
  return "";
}

function nthNonEmptyLine(text: string, n: number): string {
  let i = 0;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (i === n) return trimmed;
    i += 1;
  }
  return "";
}

function deriveCompanyName(title: string, company: string | null): string {
  // Title often looks like "VP of Product at Lattice Forge" or "VP Product på Norrsken Labs".
  const match = title.match(/\s+(?:at|på|hos|bei|chez)\s+(.+)$/i);
  if (match) return match[1].trim();
  if (company) {
    const sentence = company.split(/[.!?]/)[0]?.trim() ?? "";
    const head = sentence.split(/\s+is\s+|\s+är\s+/i)[0]?.trim();
    if (head) return head;
  }
  return "";
}

function exemplarToRow(ex: Exemplar): Row {
  const label = firstLine(ex.profile) || ex.label;
  const title = nthNonEmptyLine(ex.profile, 1) || "";
  return {
    id: ex.id,
    source: "exemplar",
    label,
    title,
    companyName: deriveCompanyName(title, ex.company),
    profile: ex.profile,
    company: ex.company,
    status: "idle",
    selected: false,
    expanded: false,
    underTheHood: false,
    thinking: "",
    result: null,
    error: null,
    confirmedAt: null,
    hoveredQuote: null,
  };
}

const INITIAL_ROWS: Row[] = EXEMPLARS.map(exemplarToRow);

function normalise(text: string): string {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}

function findQuoteSpan(input: string, quote: string): [number, number] | null {
  if (!quote.trim()) return null;
  const flatInput = normalise(input);
  const flatQuote = normalise(quote);
  if (!flatQuote || !flatInput.includes(flatQuote)) return null;
  const lower = input.toLowerCase();
  const target = flatQuote;
  let ti = 0;
  let start = -1;
  for (let i = 0; i < lower.length; i++) {
    const ch = lower[i];
    const isWs = /\s/.test(ch);
    const cmp = isWs ? " " : ch;
    if (ti < target.length && cmp === target[ti]) {
      if (ti === 0) start = i;
      ti += 1;
      if (ti === target.length) return [start, i + 1];
    } else if (target[ti] === " " && isWs) {
      // skip extra whitespace
    } else {
      if (ti > 0 && start !== -1) {
        i = start;
      }
      ti = 0;
      start = -1;
    }
  }
  return null;
}

function highlightInput(text: string, quote: string | null): ReactNode {
  if (!text) return text;
  if (!quote) return text;
  const span = findQuoteSpan(text, quote);
  if (!span) return text;
  const [start, end] = span;
  return (
    <>
      {text.slice(0, start)}
      <mark>{text.slice(start, end)}</mark>
      {text.slice(end)}
    </>
  );
}

function formatNumber(n: number, digits = 0): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function rowInputText(row: Row): string {
  return row.company ? `${row.profile}\n\n${row.company}` : row.profile;
}

function FitDimensionsView({
  dims,
}: {
  dims: EnrichOutput["fit_score"]["dimensions"];
}) {
  return (
    <div className={styles.dimensions}>
      {DIMENSION_LABELS.map(([key, label]) => {
        const v = dims[key] ?? 0;
        return (
          <div key={key} className={styles.dim}>
            <span className={styles.dimLabel}>{label}</span>
            <span className={styles.dimBar}>
              <span
                className={styles.dimFill}
                style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }}
              />
            </span>
            <span className={styles.dimValue}>{v.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

function ClaimsList({
  claims,
  inputText,
  onHover,
}: {
  claims: Claim[];
  inputText: string;
  onHover: (q: string | null) => void;
}) {
  if (!claims.length) {
    return <p className={styles.muted}>No grounded claims produced.</p>;
  }
  return (
    <div className={styles.claims}>
      {claims.map((c, idx) => {
        const span = findQuoteSpan(inputText, c.source_quote);
        const missing = !span;
        return (
          <div
            key={idx}
            className={styles.claim}
            onMouseEnter={() => onHover(c.source_quote)}
            onMouseLeave={() => onHover(null)}
            onFocus={() => onHover(c.source_quote)}
            onBlur={() => onHover(null)}
            tabIndex={0}
          >
            <p className={styles.claimText}>{c.text}</p>
            <div className={styles.claimQuoteRow}>
              <span
                className={`${styles.claimQuotePill} ${
                  missing ? styles.claimMissing : ""
                }`}
                title={c.source_quote}
              >
                {missing ? "quote not found in input" : `“${c.source_quote}”`}
              </span>
              <span className={styles.confidence}>
                conf {c.confidence.toFixed(2)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function HookView({ hook }: { hook: EnrichOutput["draft_hook"] }) {
  if (!hook?.text) return null;
  return (
    <div className={styles.hook}>
      <p className={styles.hookText}>“{hook.text}”</p>
      <p className={styles.hookMeta}>
        Uses {hook.claims_used.length} claim
        {hook.claims_used.length === 1 ? "" : "s"} · confidence{" "}
        {hook.confidence.toFixed(2)}
      </p>
    </div>
  );
}

function UnderTheHood({
  row,
  onToggle,
}: {
  row: Row;
  onToggle: () => void;
}) {
  const result = row.result;
  if (!result) return null;
  const m = result.meta;
  return (
    <div className={styles.underHood}>
      <button
        type="button"
        className={styles.underHoodToggle}
        onClick={onToggle}
        aria-expanded={row.underTheHood}
      >
        <span className={styles.underHoodCaret}>
          {row.underTheHood ? "▾" : "▸"}
        </span>
        Under the hood
      </button>
      {row.underTheHood && (
        <div className={styles.underHoodBody}>
          <div className={styles.telemetryGrid}>
            <Cell label="Latency" value={`${formatNumber(m.latency_ms)} ms`} />
            <Cell label="Tokens in" value={formatNumber(m.tokens_in)} />
            <Cell label="Tokens out" value={formatNumber(m.tokens_out)} />
            <Cell
              label="Thinking"
              value={m.thinking_tokens ? formatNumber(m.thinking_tokens) : "—"}
            />
            <Cell label="Cache hit" value={m.cache_hit ? "yes" : "no"} />
            <Cell label="Model" value={m.model} />
          </div>
          <div className={styles.subPanel}>
            <p className={styles.subPanelLabel}>Eval neighbours</p>
            {m.eval_neighbours.length === 0 ? (
              <p className={styles.muted}>
                Test set not seeded yet (Phase 0b/4).
              </p>
            ) : (
              <div className={styles.neighbours}>
                {m.eval_neighbours.map((n) => (
                  <div key={n.id} className={styles.neighbourRow}>
                    <span>#{n.id}</span>
                    <span>sim {n.similarity.toFixed(2)}</span>
                    <span>
                      fit {n.score === null ? "—" : n.score.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className={styles.subPanel}>
            <p className={styles.subPanelLabel}>Reasoning trace</p>
            <pre className={styles.reasoning}>
              {row.thinking || "(no reasoning trace returned)"}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.telemetryCell}>
      <span className={styles.telemetryCellLabel}>{label}</span>
      <span className={styles.telemetryCellValue}>{value}</span>
    </div>
  );
}

function ScoreCell({ row }: { row: Row }) {
  if (row.status === "streaming") {
    return <span className={styles.scoreSpinner}>…</span>;
  }
  if (row.error) {
    return <span className={styles.scoreError}>error</span>;
  }
  if (!row.result) {
    return <span className={styles.scorePlaceholder}>—</span>;
  }
  return (
    <span className={styles.scoreValue}>
      {row.result.fit_score.value.toFixed(2)}
    </span>
  );
}

function ActionChip({ row }: { row: Row }) {
  if (row.status === "streaming") {
    return <span className={styles.actionPlaceholder}>running…</span>;
  }
  if (row.error) {
    return (
      <span className={styles.actionPlaceholder} title={row.error.message}>
        {row.error.code}
      </span>
    );
  }
  if (!row.result) {
    return <span className={styles.actionPlaceholder}>—</span>;
  }
  if (row.confirmedAt) {
    return (
      <span
        className={`${styles.actionChip} ${styles.actionChipConfirmed}`}
        title={ACTION_LABELS[row.result.action]}
      >
        ✓ Confirmed
      </span>
    );
  }
  return (
    <span
      className={`${styles.actionChip} ${
        styles[`actionChip_${row.result.action}`] ?? ""
      }`}
    >
      {ACTION_CHIP_LABELS[row.result.action]}
    </span>
  );
}

function SkeletonDrawer({ row }: { row: Row }) {
  const inputText = rowInputText(row);
  return (
    <div className={styles.drawer}>
      <div className={styles.drawerGrid}>
        <section className={styles.drawerLeft}>
          <div className={styles.chips}>
            <span className={`${styles.skelChip} ${styles.skel}`} style={{ width: "7rem" }} />
            <span className={`${styles.skelChip} ${styles.skel}`} style={{ width: "11rem" }} />
            <span className={`${styles.skelChip} ${styles.skel}`} style={{ width: "7rem" }} />
            <span className={`${styles.skelChip} ${styles.skel}`} style={{ width: "8rem" }} />
          </div>
          <div className={styles.fitRow}>
            <span className={`${styles.skel} ${styles.skelFit}`} />
            <span className={styles.fitOutOf}>fit score / 1.00</span>
          </div>
          <div className={styles.dimensions}>
            {DIMENSION_LABELS.map(([, label]) => (
              <div key={label} className={styles.dim}>
                <span className={styles.dimLabel}>{label}</span>
                <span className={styles.dimBar} />
                <span className={`${styles.skel} ${styles.skelDimValue}`} />
              </div>
            ))}
          </div>
          <div className={styles.claimsHeader}>
            <span className={styles.sectionLabel}>Claims</span>
          </div>
          <div className={styles.claims}>
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className={styles.skelClaim}>
                <span className={`${styles.skel} ${styles.skelClaimLine}`} />
                <span className={`${styles.skel} ${styles.skelClaimQuote}`} />
              </div>
            ))}
          </div>
          <div className={styles.hook}>
            <span className={`${styles.skel} ${styles.skelHookLine}`} />
            <span
              className={`${styles.skel} ${styles.skelHookLine}`}
              style={{ width: "65%" }}
            />
          </div>
          <div className={styles.actionRow}>
            <span className={styles.actionMeta}>Recommended action</span>
            <span className={`${styles.skel} ${styles.skelActionButton}`} />
          </div>
        </section>
        <aside className={styles.drawerRight}>
          <span className={styles.sectionLabel}>Source</span>
          <div className={styles.inputView}>{inputText}</div>
        </aside>
      </div>
    </div>
  );
}

function RowDrawer({
  row,
  onAct,
  onHover,
  onToggleUnderHood,
}: {
  row: Row;
  onAct: () => void;
  onHover: (q: string | null) => void;
  onToggleUnderHood: () => void;
}) {
  const result = row.result;
  if (row.error) {
    return (
      <div className={styles.drawer}>
        <div className={styles.errorBanner}>
          <strong>{row.error.code}</strong> · {row.error.message}
        </div>
      </div>
    );
  }
  if (!result) {
    if (row.status === "streaming") {
      return <SkeletonDrawer row={row} />;
    }
    return (
      <div className={styles.drawer}>
        <p className={styles.muted}>
          Run this row to see the structured output.
        </p>
      </div>
    );
  }
  const c = result.classification;
  const fit = result.fit_score;
  const inputText = rowInputText(row);
  return (
    <div className={styles.drawer}>
      <div className={styles.drawerGrid}>
        <section className={styles.drawerLeft}>
          <div className={styles.chips}>
            <span className={styles.chip}>{c.industry}</span>
            <span className={styles.chip}>{c.segment}</span>
            <span className={`${styles.chip} ${styles.chipMuted}`}>
              Seniority · {c.seniority}
            </span>
            <span className={`${styles.chip} ${styles.chipMuted}`}>
              Headcount · {c.company_size}
            </span>
          </div>
          <div className={styles.fitRow}>
            <span className={styles.fitValue}>{fit.value.toFixed(2)}</span>
            <span className={styles.fitOutOf}>fit score / 1.00</span>
          </div>
          <FitDimensionsView dims={fit.dimensions} />
          <div className={styles.claimsHeader}>
            <span className={styles.sectionLabel}>Claims</span>
          </div>
          <ClaimsList
            claims={result.claims}
            inputText={inputText}
            onHover={onHover}
          />
          <HookView hook={result.draft_hook} />
          <div className={styles.actionRow}>
            <span className={styles.actionMeta}>
              Recommended · <strong>{result.action.replace("_", " ")}</strong>
            </span>
            <button
              type="button"
              className={`${styles.actionButton} ${
                result.action === "discard" || result.action === "refuse"
                  ? styles.actionButtonMuted
                  : ""
              }`}
              onClick={onAct}
              disabled={Boolean(row.confirmedAt)}
            >
              {row.confirmedAt ? "Confirmed" : ACTION_LABELS[result.action]}
            </button>
          </div>
        </section>
        <aside className={styles.drawerRight}>
          <span className={styles.sectionLabel}>
            Source{" "}
            {row.hoveredQuote && (
              <span className={styles.muted}>· highlighting quote</span>
            )}
          </span>
          <div className={styles.inputView}>
            {highlightInput(inputText, row.hoveredQuote)}
          </div>
        </aside>
      </div>
      <UnderTheHood row={row} onToggle={onToggleUnderHood} />
    </div>
  );
}

function QueueRow({
  row,
  onToggleSelect,
  onToggleExpand,
  onAct,
  onHover,
  onToggleUnderHood,
  disableSelection,
}: {
  row: Row;
  onToggleSelect: () => void;
  onToggleExpand: () => void;
  onAct: () => void;
  onHover: (q: string | null) => void;
  onToggleUnderHood: () => void;
  disableSelection: boolean;
}) {
  return (
    <div
      className={`${styles.row} ${row.expanded ? styles.rowExpanded : ""} ${
        row.confirmedAt ? styles.rowConfirmed : ""
      }`}
      data-status={row.status}
    >
      <div className={styles.rowSummary}>
        <input
          type="checkbox"
          className={styles.rowCheck}
          checked={row.selected}
          onChange={onToggleSelect}
          disabled={disableSelection || row.status === "streaming"}
          aria-label={`Select ${row.label}`}
        />
        <div className={styles.rowIdent}>
          <span className={styles.rowName}>{row.label}</span>
          {row.title && (
            <span className={styles.rowTitle}>{row.title}</span>
          )}
        </div>
        <span className={styles.rowCompany}>
          {row.companyName || (row.source === "paste" ? "—" : "")}
        </span>
        <span className={styles.rowScore}>
          <ScoreCell row={row} />
        </span>
        <span className={styles.rowAction}>
          <ActionChip row={row} />
        </span>
        <button
          type="button"
          className={styles.rowExpandBtn}
          onClick={onToggleExpand}
          aria-expanded={row.expanded}
          aria-label={row.expanded ? "Collapse row" : "Expand row"}
        >
          {row.expanded ? "▾" : "▸"}
        </button>
      </div>
      {row.expanded && (
        <RowDrawer
          row={row}
          onAct={onAct}
          onHover={onHover}
          onToggleUnderHood={onToggleUnderHood}
        />
      )}
    </div>
  );
}

function Composer({
  onAdd,
  disabled,
}: {
  onAdd: (profile: string, company: string | null) => void;
  disabled: boolean;
}) {
  const [profile, setProfile] = useState("");
  const [company, setCompany] = useState("");
  const profileOver = profile.length > PROFILE_CAP;
  const companyOver = company.length > COMPANY_CAP;
  const canAdd =
    profile.trim().length > 0 && !profileOver && !companyOver && !disabled;
  return (
    <details className={styles.composer}>
      <summary className={styles.composerSummary}>
        Add your own lead
      </summary>
      <div className={styles.composerBody}>
        <div className={styles.field}>
          <label htmlFor="composer-profile">
            <span>Profile (required)</span>
            <span className={profileOver ? styles.fieldOver : undefined}>
              {profile.length} / {PROFILE_CAP}
            </span>
          </label>
          <textarea
            id="composer-profile"
            value={profile}
            onChange={(e) => setProfile(e.target.value)}
            placeholder="Paste a LinkedIn-style profile…"
          />
        </div>
        <div className={styles.field}>
          <label htmlFor="composer-company">
            <span>Company (optional)</span>
            <span className={companyOver ? styles.fieldOver : undefined}>
              {company.length} / {COMPANY_CAP}
            </span>
          </label>
          <textarea
            id="composer-company"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            placeholder="Paste a company description…"
          />
        </div>
        <div className={styles.composerRow}>
          <p className={styles.muted}>Inputs are not stored.</p>
          <button
            type="button"
            className={styles.composerButton}
            disabled={!canAdd}
            onClick={() => {
              onAdd(profile, company.trim() ? company : null);
              setProfile("");
              setCompany("");
            }}
          >
            Add to queue
          </button>
        </div>
      </div>
    </details>
  );
}

export default function IntegratedPage() {
  const [rows, setRows] = useState<Row[]>(INITIAL_ROWS);
  const abortControllers = useRef<Map<string, AbortController>>(new Map());
  const pasteCounter = useRef(1);

  const updateRow = useCallback(
    (id: string, patch: Partial<Row> | ((r: Row) => Partial<Row>)) => {
      setRows((prev) =>
        prev.map((r) => {
          if (r.id !== id) return r;
          const p = typeof patch === "function" ? patch(r) : patch;
          return { ...r, ...p };
        }),
      );
    },
    [],
  );

  const streamRow = useCallback(
    async (rowSnapshot: Row) => {
      abortControllers.current.get(rowSnapshot.id)?.abort();
      const controller = new AbortController();
      abortControllers.current.set(rowSnapshot.id, controller);

      updateRow(rowSnapshot.id, {
        status: "streaming",
        thinking: "",
        result: null,
        error: null,
        confirmedAt: null,
        hoveredQuote: null,
      });

      let terminal = false;
      try {
        const res = await fetch("/api/enrich", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            profile: rowSnapshot.profile,
            company: rowSnapshot.company,
            example_id:
              rowSnapshot.source === "exemplar" ? rowSnapshot.id : null,
          }),
          signal: controller.signal,
        });
        if (!res.ok || !res.body) {
          updateRow(rowSnapshot.id, {
            status: "error",
            selected: false,
            error: {
              code: `http_${res.status}`,
              message: `Upstream returned ${res.status}.`,
            },
          });
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
                const event = JSON.parse(json) as StreamEvent;
                if (event.type === "thinking") {
                  updateRow(rowSnapshot.id, (r) => ({
                    thinking: r.thinking + event.delta,
                  }));
                } else if (event.type === "result") {
                  updateRow(rowSnapshot.id, {
                    result: event.output,
                    status: "done",
                    selected: false,
                  });
                  terminal = true;
                } else if (event.type === "error") {
                  updateRow(rowSnapshot.id, {
                    status: "error",
                    selected: false,
                    error: { code: event.code, message: event.message },
                  });
                  terminal = true;
                }
              } catch {
                // ignore malformed frame
              }
            }
            sep = buffer.indexOf("\n\n");
          }
        }
        if (!terminal) {
          updateRow(rowSnapshot.id, {
            status: "error",
            selected: false,
            error: {
              code: "stream_ended",
              message: "Stream ended without a result.",
            },
          });
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        updateRow(rowSnapshot.id, {
          status: "error",
          selected: false,
          error: {
            code: "network",
            message: (err as Error).message ?? "Network error",
          },
        });
      } finally {
        abortControllers.current.delete(rowSnapshot.id);
      }
    },
    [updateRow],
  );

  const onToggleSelect = useCallback((id: string) => {
    setRows((prev) =>
      prev.map((r) =>
        r.id === id && r.status !== "streaming"
          ? { ...r, selected: !r.selected }
          : r,
      ),
    );
  }, []);

  const onToggleExpand = useCallback((id: string) => {
    setRows((prev) =>
      prev.map((r) => (r.id === id ? { ...r, expanded: !r.expanded } : r)),
    );
  }, []);

  const onToggleUnderHood = useCallback((id: string) => {
    setRows((prev) =>
      prev.map((r) =>
        r.id === id ? { ...r, underTheHood: !r.underTheHood } : r,
      ),
    );
  }, []);

  const onHover = useCallback(
    (id: string, quote: string | null) => {
      updateRow(id, { hoveredQuote: quote });
    },
    [updateRow],
  );

  const onConfirm = useCallback(
    (id: string) => {
      updateRow(id, { confirmedAt: Date.now() });
    },
    [updateRow],
  );

  const onAddPaste = useCallback(
    (profile: string, company: string | null) => {
      const idx = pasteCounter.current++;
      const id = `paste-${idx}`;
      const label = firstLine(profile).slice(0, 60) || `Paste ${idx}`;
      const title = nthNonEmptyLine(profile, 1).slice(0, 80);
      const newRow: Row = {
        id,
        source: "paste",
        label,
        title,
        companyName: deriveCompanyName(title, company),
        profile,
        company,
        status: "idle",
        selected: true,
        expanded: false,
        underTheHood: false,
        thinking: "",
        result: null,
        error: null,
        confirmedAt: null,
        hoveredQuote: null,
      };
      setRows((prev) => [...prev, newRow]);
    },
    [],
  );

  const selectedIdleRows = useMemo(
    () =>
      rows.filter(
        (r) => r.selected && r.status !== "streaming" && r.status !== "done",
      ),
    [rows],
  );
  const anyStreaming = useMemo(
    () => rows.some((r) => r.status === "streaming"),
    [rows],
  );
  const anySelected = useMemo(() => rows.some((r) => r.selected), [rows]);

  const onRunSelected = useCallback(() => {
    if (anyStreaming) return;
    const targets = rows.filter((r) => r.selected);
    for (const target of targets) {
      void streamRow(target);
    }
  }, [rows, streamRow, anyStreaming]);

  const onSelectAll = useCallback(() => {
    setRows((prev) =>
      prev.map((r) => (r.status === "streaming" ? r : { ...r, selected: true })),
    );
  }, []);

  const onClearSelection = useCallback(() => {
    setRows((prev) => prev.map((r) => ({ ...r, selected: false })));
  }, []);

  useEffect(() => {
    const controllers = abortControllers.current;
    return () => {
      for (const c of controllers.values()) c.abort();
    };
  }, []);

  return (
    <main>
      <h1 className={styles.heroTitle}>Triage new leads against your ICP</h1>
      <p className={styles.heroLede}>
        Each lead gets a fit score, a set of grounded claims, a drafted
        outreach hook, and a recommended action. Select leads, run them, and
        expand a row for the full breakdown.
      </p>

      <div className={styles.queue}>
        <div className={styles.queueHeader}>
          <div className={styles.queueHeaderLeft}>
            <button
              type="button"
              className={styles.linkButton}
              onClick={anySelected ? onClearSelection : onSelectAll}
            >
              {anySelected ? "Clear selection" : "Select all"}
            </button>
            <span className={styles.muted}>
              {selectedIdleRows.length} selected, ready to run
            </span>
          </div>
          <button
            type="button"
            className={styles.runButton}
            onClick={onRunSelected}
            disabled={selectedIdleRows.length === 0 || anyStreaming}
          >
            {anyStreaming ? "Running…" : `Run ${selectedIdleRows.length || ""}`.trim()}
          </button>
        </div>

        <div className={styles.queueColumnHeader} aria-hidden="true">
          <span />
          <span>Lead</span>
          <span>Company</span>
          <span>Fit</span>
          <span>Action</span>
          <span />
        </div>

        <div className={styles.queueList}>
          {rows.map((row) => (
            <QueueRow
              key={row.id}
              row={row}
              disableSelection={anyStreaming}
              onToggleSelect={() => onToggleSelect(row.id)}
              onToggleExpand={() => onToggleExpand(row.id)}
              onAct={() => onConfirm(row.id)}
              onHover={(q) => onHover(row.id, q)}
              onToggleUnderHood={() => onToggleUnderHood(row.id)}
            />
          ))}
        </div>

        <Composer onAdd={onAddPaste} disabled={anyStreaming} />
      </div>
    </main>
  );
}
