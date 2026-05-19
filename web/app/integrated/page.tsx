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
import {
  COMPANY_CAP,
  PROFILE_CAP,
  deriveCompanyName,
  exemplarToLead,
} from "@/lib/leads";
import { parseSseStream } from "@/lib/sse";
import { track } from "@/lib/analytics";
import { LowSimilarityBadge } from "@/app/_shared/low-similarity-badge";
import { QueueEmpty } from "@/app/_shared/queue-empty";
import type {
  Action,
  Claim,
  EnrichOutput,
  StreamEvent,
} from "@/lib/types";
import { firstLine, formatNumber, nthNonEmptyLine } from "@/lib/utils";
import styles from "./integrated.module.css";

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

type ActionFilter = "auto_add" | "propose" | "discard" | "refuse" | "unscored";
type SortBy = "recent" | "fit" | "name" | "action";

const ACTION_FILTER_ORDER: ActionFilter[] = [
  "auto_add",
  "propose",
  "discard",
  "refuse",
  "unscored",
];

const ACTION_FILTER_LABELS: Record<ActionFilter, string> = {
  auto_add: "Auto-add",
  propose: "Propose",
  discard: "Discard",
  refuse: "Refuse",
  unscored: "Unscored",
};

const SORT_LABELS: Record<SortBy, string> = {
  recent: "Most recent",
  fit: "Fit score",
  name: "Name",
  action: "Action",
};

type RowStatus = "idle" | "streaming" | "done" | "error";

type ErrorState = { code: string; message: string };

interface Row {
  id: string;
  source: "exemplar" | "paste";
  leadName: string;
  title: string;
  companyName: string;
  profile: string;
  company: string | null;
  status: RowStatus;
  selected: boolean;
  underTheHood: boolean;
  thinking: string;
  result: EnrichOutput | null;
  error: ErrorState | null;
  confirmedAt: number | null;
  hoveredQuote: string | null;
  createdAt: number;
}

function exemplarToRow(ex: Exemplar, createdAt: number): Row {
  return {
    ...exemplarToLead(ex),
    status: "idle",
    selected: false,
    underTheHood: false,
    thinking: "",
    result: null,
    error: null,
    confirmedAt: null,
    hoveredQuote: null,
    createdAt,
  };
}

const INITIAL_ROWS: Row[] = (() => {
  const now = Date.now();
  return EXEMPLARS.map((ex, i) => exemplarToRow(ex, now - i));
})();

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
  onRerunLive,
}: {
  row: Row;
  onToggle: () => void;
  onRerunLive: () => void;
}) {
  const result = row.result;
  if (!result) return null;
  const m = result.meta;
  return (
    <div className={styles.underHood}>
      <div className={styles.underHoodHeader}>
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
        {m.snapshot_served && (
          <button
            type="button"
            className={styles.rerunLive}
            onClick={onRerunLive}
            disabled={row.status === "streaming"}
            title="Bypass the cached snapshot and run this exemplar against the live Anthropic API."
          >
            Re-run live
          </button>
        )}
      </div>
      {row.underTheHood && (
        <div className={styles.underHoodBody}>
          <div className={styles.telemetryGrid}>
            <Cell label="Latency" value={`${formatNumber(m.latency_ms)} ms`} />
            <Cell label="Tokens in" value={formatNumber(m.tokens_in)} />
            <Cell label="Tokens out" value={formatNumber(m.tokens_out)} />
            <Cell
              label="Thinking"
              value={
                m.thinking_tokens
                  ? `${formatNumber(m.thinking_tokens)} tokens`
                  : `${formatNumber(m.thinking_budget)} budget`
              }
            />
            <Cell label="Cache hit" value={m.cache_hit ? "yes" : "no"} />
            <Cell label="Model" value={m.model} />
          </div>
          <div className={styles.subPanel}>
            <p className={styles.subPanelLabel}>Eval neighbours</p>
            {m.eval_neighbours.length === 0 ? (
              <p className={styles.muted}>
                Test set not seeded yet.
              </p>
            ) : (
              <>
                <LowSimilarityBadge
                  neighbours={m.eval_neighbours}
                  className={styles.lowSimilarityBadge}
                />
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
              </>
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

function SourcePane({
  profile,
  company,
  hoveredQuote = null,
}: {
  profile: string;
  company: string | null;
  hoveredQuote?: string | null;
}) {
  return (
    <section className={styles.sourcePane}>
      <span className={styles.sectionLabel}>
        Source{" "}
        {hoveredQuote && (
          <span className={styles.muted}>· highlighting quote</span>
        )}
      </span>
      <div className={styles.sourceSection}>
        <span className={styles.sourceSectionLabel}>Profile</span>
        <div className={styles.inputView}>
          {highlightInput(profile, hoveredQuote)}
        </div>
      </div>
      {company !== null && (
        <div className={styles.sourceSection}>
          <span className={styles.sourceSectionLabel}>Company</span>
          <div className={styles.inputView}>
            {highlightInput(company, hoveredQuote)}
          </div>
        </div>
      )}
    </section>
  );
}

function SkeletonBody({ row }: { row: Row }) {
  return (
    <>
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
      <SourcePane profile={row.profile} company={row.company} />
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
    </>
  );
}

function SidepanelBody({
  row,
  onHover,
  onToggleUnderHood,
  onRerunLive,
}: {
  row: Row;
  onHover: (q: string | null) => void;
  onToggleUnderHood: () => void;
  onRerunLive: () => void;
}) {
  if (row.error) {
    return (
      <div className={styles.errorBanner}>
        <strong>{row.error.code}</strong> · {row.error.message}
      </div>
    );
  }
  const result = row.result;
  if (!result) {
    if (row.status === "streaming") {
      return <SkeletonBody row={row} />;
    }
    return (
      <>
        <p className={styles.muted}>
          Select this lead and press Run to score it.
        </p>
        <SourcePane profile={row.profile} company={row.company} />
      </>
    );
  }
  const c = result.classification;
  const fit = result.fit_score;
  const inputText = rowInputText(row);
  return (
    <>
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
      <SourcePane
        profile={row.profile}
        company={row.company}
        hoveredQuote={row.hoveredQuote}
      />
      <div className={styles.claimsHeader}>
        <span className={styles.sectionLabel}>Claims</span>
      </div>
      <ClaimsList
        claims={result.claims}
        inputText={inputText}
        onHover={onHover}
      />
      <HookView hook={result.draft_hook} />
      <UnderTheHood
        row={row}
        onToggle={onToggleUnderHood}
        onRerunLive={onRerunLive}
      />
    </>
  );
}

function Sidepanel({
  row,
  onClose,
  onAct,
  onHover,
  onToggleUnderHood,
  onRerunLive,
}: {
  row: Row | null;
  onClose: () => void;
  onAct: () => void;
  onHover: (q: string | null) => void;
  onToggleUnderHood: () => void;
  onRerunLive: () => void;
}) {
  const open = row !== null;
  const result = row?.result ?? null;
  return (
    <aside
      className={`${styles.sidepanel} ${open ? styles.sidepanelOpen : ""}`}
      aria-hidden={!open}
      aria-label="Lead details"
    >
      {row && (
        <>
          <header className={styles.sidepanelHeader}>
            <div className={styles.sidepanelTitle}>
              <h2 className={styles.sidepanelName}>{row.leadName}</h2>
              {row.title && (
                <p className={styles.sidepanelSubtitle}>{row.title}</p>
              )}
              {row.companyName && (
                <p className={styles.sidepanelCompany}>{row.companyName}</p>
              )}
            </div>
            <button
              type="button"
              className={styles.sidepanelClose}
              onClick={onClose}
              aria-label="Close panel"
            >
              ✕
            </button>
          </header>
          <div className={styles.sidepanelBody}>
            <SidepanelBody
              row={row}
              onHover={onHover}
              onToggleUnderHood={onToggleUnderHood}
              onRerunLive={onRerunLive}
            />
          </div>
          {result && !row.error && (
            <footer className={styles.sidepanelFooter}>
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
            </footer>
          )}
        </>
      )}
    </aside>
  );
}

function QueueRow({
  row,
  isSelected,
  onToggleSelect,
  onSelectRow,
  disableSelection,
}: {
  row: Row;
  isSelected: boolean;
  onToggleSelect: () => void;
  onSelectRow: () => void;
  disableSelection: boolean;
}) {
  return (
    <div
      className={`${styles.row} ${isSelected ? styles.rowSelected : ""} ${
        row.confirmedAt ? styles.rowConfirmed : ""
      }`}
      data-status={row.status}
    >
      <div
        className={styles.rowSummary}
        onClick={() => {
          if (row.source === "exemplar" && !isSelected) {
            track({ name: "example-loaded", props: { surface: "integrated", exampleId: row.id } });
          }
          onSelectRow();
        }}
        role="button"
        tabIndex={0}
        aria-pressed={isSelected}
        aria-label={`View details for ${row.leadName}`}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            if (row.source === "exemplar" && !isSelected) {
              track({ name: "example-loaded", props: { surface: "integrated", exampleId: row.id } });
            }
            onSelectRow();
          }
        }}
      >
        <input
          type="checkbox"
          className={styles.rowCheck}
          checked={row.selected}
          onChange={onToggleSelect}
          onClick={(e) => e.stopPropagation()}
          disabled={disableSelection || row.status === "streaming"}
          aria-label={`Select ${row.leadName}`}
        />
        <div className={styles.rowIdent}>
          <span className={styles.rowName}>{row.leadName}</span>
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
          {row.result?.meta.snapshot_served && (
            <span
              className={styles.snapshotPill}
              title="Response served from a committed exemplar snapshot. Open the row and use Re-run live to hit the live API."
            >
              Cached
            </span>
          )}
        </span>
      </div>
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
        <span className={styles.composerToggle}>Add your own lead</span>
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
              track({ name: "own-input-pasted", props: { surface: "integrated" } });
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
  const lastLowFitRef = useRef<number | null>(null);

  useEffect(() => {
    const onHide = () => {
      if (lastLowFitRef.current !== null) {
        track({
          name: "page-bounce-after-low-fit",
          props: { fitScore: lastLowFitRef.current },
        });
      }
    };
    window.addEventListener("pagehide", onHide);
    return () => window.removeEventListener("pagehide", onHide);
  }, []);

  const [searchQuery, setSearchQuery] = useState("");
  const [actionFilters, setActionFilters] = useState<Set<ActionFilter>>(
    () => new Set(ACTION_FILTER_ORDER),
  );
  const [sortBy, setSortBy] = useState<SortBy>("recent");
  const [selectedRowId, setSelectedRowId] = useState<string | null>(null);

  const selectedRow = useMemo(
    () => rows.find((r) => r.id === selectedRowId) ?? null,
    [rows, selectedRowId],
  );

  useEffect(() => {
    if (selectedRowId !== null && !selectedRow) {
      setSelectedRowId(null);
    }
  }, [selectedRow, selectedRowId]);

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
    async (rowSnapshot: Row, opts: { bypassCache?: boolean } = {}) => {
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
            bypass_cache: opts.bypassCache === true,
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
        await parseSseStream<StreamEvent>(res, (event) => {
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
            const fit = event.output.fit_score.value;
            track({
              name: "completion",
              props: {
                surface: "integrated",
                action: event.output.action,
                fitScore: fit,
              },
            });
            if (fit < 0.5) lastLowFitRef.current = fit;
          } else if (event.type === "error") {
            updateRow(rowSnapshot.id, {
              status: "error",
              selected: false,
              error: { code: event.code, message: event.message },
            });
            terminal = true;
            track({ name: "error", props: { surface: "integrated", kind: event.code } });
          }
        });
        if (!terminal) {
          updateRow(rowSnapshot.id, {
            status: "error",
            selected: false,
            error: {
              code: "stream_ended",
              message: "Stream ended without a result.",
            },
          });
          track({ name: "error", props: { surface: "integrated", kind: "stream_ended" } });
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
        track({ name: "error", props: { surface: "integrated", kind: "network" } });
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

  const onSelectRow = useCallback((id: string) => {
    setSelectedRowId((prev) => (prev === id ? null : id));
  }, []);

  const onCloseSidepanel = useCallback(() => {
    setSelectedRowId(null);
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
      const leadName = firstLine(profile).slice(0, 60) || `Paste ${idx}`;
      const title = nthNonEmptyLine(profile, 1).slice(0, 80);
      const newRow: Row = {
        id,
        source: "paste",
        leadName,
        title,
        companyName: deriveCompanyName(title, company),
        profile,
        company,
        status: "idle",
        selected: true,
        underTheHood: false,
        thinking: "",
        result: null,
        error: null,
        confirmedAt: null,
        hoveredQuote: null,
        createdAt: Date.now(),
      };
      setRows((prev) => [newRow, ...prev]);
    },
    [],
  );

  const visibleRows = useMemo<Row[]>(() => {
    const q = searchQuery.trim().toLowerCase();
    const matchesQuery = (r: Row) =>
      !q ||
      r.leadName.toLowerCase().includes(q) ||
      r.title.toLowerCase().includes(q) ||
      r.companyName.toLowerCase().includes(q);

    const matchesAction = (r: Row): boolean => {
      if (actionFilters.size === 0) return false;
      if (!r.result) return actionFilters.has("unscored");
      return actionFilters.has(r.result.action as ActionFilter);
    };

    const filtered = rows.filter((r) => matchesQuery(r) && matchesAction(r));

    if (sortBy === "name") {
      return [...filtered].sort((a, b) =>
        a.leadName.localeCompare(b.leadName, undefined, { sensitivity: "base" }),
      );
    }
    if (sortBy === "fit") {
      return [...filtered].sort((a, b) => {
        const av = a.result?.fit_score.value ?? -1;
        const bv = b.result?.fit_score.value ?? -1;
        if (av !== bv) return bv - av;
        return b.createdAt - a.createdAt;
      });
    }
    if (sortBy === "action") {
      const idx = (r: Row): number => {
        if (!r.result) return ACTION_FILTER_ORDER.indexOf("unscored");
        return ACTION_FILTER_ORDER.indexOf(r.result.action as ActionFilter);
      };
      return [...filtered].sort((a, b) => {
        const sd = idx(a) - idx(b);
        if (sd !== 0) return sd;
        return b.createdAt - a.createdAt;
      });
    }
    return [...filtered].sort((a, b) => b.createdAt - a.createdAt);
  }, [rows, searchQuery, actionFilters, sortBy]);

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
  const selectedCount = useMemo(
    () => rows.reduce((acc, r) => (r.selected ? acc + 1 : acc), 0),
    [rows],
  );

  const onRunSelected = useCallback(() => {
    if (anyStreaming) return;
    const targets = rows.filter((r) => r.selected);
    for (const target of targets) {
      void streamRow(target);
    }
  }, [rows, streamRow, anyStreaming]);

  const onSelectAll = useCallback(() => {
    const visibleIds = new Set(visibleRows.map((r) => r.id));
    setRows((prev) =>
      prev.map((r) =>
        visibleIds.has(r.id) && r.status !== "streaming"
          ? { ...r, selected: true }
          : r,
      ),
    );
  }, [visibleRows]);

  const onClearSelection = useCallback(() => {
    setRows((prev) => prev.map((r) => ({ ...r, selected: false })));
  }, []);

  const onToggleActionFilter = useCallback((f: ActionFilter) => {
    setActionFilters((prev) => {
      const next = new Set(prev);
      if (next.has(f)) next.delete(f);
      else next.add(f);
      return next;
    });
  }, []);

  const onResetFilters = useCallback(() => {
    setSearchQuery("");
    setActionFilters(new Set(ACTION_FILTER_ORDER));
  }, []);

  const onDeleteSelected = useCallback(() => {
    const toRemove = new Set(
      rows.filter((r) => r.selected).map((r) => r.id),
    );
    if (toRemove.size === 0) return;
    for (const id of toRemove) {
      abortControllers.current.get(id)?.abort();
      abortControllers.current.delete(id);
    }
    if (selectedRowId && toRemove.has(selectedRowId)) {
      setSelectedRowId(null);
    }
    setRows((prev) => prev.filter((r) => !toRemove.has(r.id)));
  }, [rows, selectedRowId]);

  useEffect(() => {
    const controllers = abortControllers.current;
    return () => {
      for (const c of controllers.values()) c.abort();
    };
  }, []);

  useEffect(() => {
    if (selectedRowId === null) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setSelectedRowId(null);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedRowId]);

  const panelOpen = selectedRow !== null;

  return (
    <main className={panelOpen ? styles.mainWithPanel : undefined}>
      <h1 className={styles.heroTitle}>Triage new leads against your ICP</h1>
      <p className={styles.heroLede}>
        Each lead gets a fit score, a set of grounded claims, a drafted
        outreach hook, and a recommended action. Select leads, run them, and
        click a row to open the full breakdown in the side panel.
      </p>

      <div className={styles.queue}>
        <div className={styles.queueHeader}>
          <div className={styles.queueHeaderLeft}>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={anySelected ? onClearSelection : onSelectAll}
            >
              {anySelected ? "Clear selection" : "Select all visible"}
            </button>
            <span className={styles.muted}>
              {selectedIdleRows.length} selected, ready to run
            </span>
            {selectedCount > 0 && (
              <button
                type="button"
                className={styles.deleteButton}
                onClick={onDeleteSelected}
              >
                Delete {selectedCount}
              </button>
            )}
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

        <div className={styles.queueControls}>
          <div className={styles.queueControlsRow}>
            <input
              type="search"
              className={styles.searchInput}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by name, title, company…"
              aria-label="Search leads"
            />
            <select
              className={styles.sortSelect}
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortBy)}
              aria-label="Sort leads"
            >
              {(Object.keys(SORT_LABELS) as SortBy[]).map((s) => (
                <option key={s} value={s}>
                  Sort: {SORT_LABELS[s]}
                </option>
              ))}
            </select>
            <span className={styles.visibleCount}>
              {visibleRows.length === rows.length
                ? `${rows.length} lead${rows.length === 1 ? "" : "s"}`
                : `${visibleRows.length} of ${rows.length} leads`}
            </span>
          </div>
          <div
            className={styles.statusChips}
            role="group"
            aria-label="Filter by action"
          >
            {ACTION_FILTER_ORDER.map((f) => {
              const active = actionFilters.has(f);
              return (
                <button
                  key={f}
                  type="button"
                  className={`${styles.statusChip} ${
                    active ? styles.statusChipActive : ""
                  }`}
                  onClick={() => onToggleActionFilter(f)}
                  aria-pressed={active}
                >
                  {ACTION_FILTER_LABELS[f]}
                </button>
              );
            })}
          </div>
        </div>

        <div className={styles.queueColumnHeader} aria-hidden="true">
          <span />
          <span>Lead</span>
          <span>Company</span>
          <span>Fit</span>
          <span>Action</span>
        </div>

        {visibleRows.length === 0 ? (
          <QueueEmpty
            query={searchQuery}
            onReset={onResetFilters}
            className={styles.queueEmpty}
            buttonClassName={styles.linkButton}
          />
        ) : (
          <div className={styles.queueList}>
            {visibleRows.map((row) => (
              <QueueRow
                key={row.id}
                row={row}
                isSelected={row.id === selectedRowId}
                disableSelection={anyStreaming}
                onToggleSelect={() => onToggleSelect(row.id)}
                onSelectRow={() => onSelectRow(row.id)}
              />
            ))}
          </div>
        )}

        <Composer onAdd={onAddPaste} disabled={anyStreaming} />
      </div>

      <Sidepanel
        row={selectedRow}
        onClose={onCloseSidepanel}
        onAct={() => selectedRow && onConfirm(selectedRow.id)}
        onHover={(q) => selectedRow && onHover(selectedRow.id, q)}
        onToggleUnderHood={() =>
          selectedRow && onToggleUnderHood(selectedRow.id)
        }
        onRerunLive={() => {
          if (selectedRow) streamRow(selectedRow, { bypassCache: true });
        }}
      />
    </main>
  );
}
