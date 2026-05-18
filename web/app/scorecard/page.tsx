import Link from "next/link";

import {
  fmtMs,
  fmtNumber,
  fmtPercent,
  loadAnnotations,
  loadSnapshot,
  type Annotation,
  type ModeBlock,
  type Snapshot,
} from "@/lib/scorecard";

import styles from "./scorecard.module.css";

// Snapshot updates nightly. 5-minute ISR keeps Vercel function invocations
// rare without making a stale snapshot visible for long.
export const revalidate = 300;

export default function ScorecardPage() {
  const snapshot = loadSnapshot();
  const annotations = loadAnnotations();
  if (!snapshot) {
    return <EmptyState />;
  }
  return <Scorecard snapshot={snapshot} annotations={annotations} />;
}

function EmptyState() {
  return (
    <main>
      <h1>Eval scorecard</h1>
      <p style={{ color: "var(--muted)", maxWidth: "60ch" }}>
        No eval snapshot has been committed yet. The nightly cron writes
        `data/eval_runs/latest.json` at 02:00 UTC. Until the first run lands,
        the methodology page describes what will appear here.
      </p>
    </main>
  );
}

function Scorecard({
  snapshot,
  annotations,
}: {
  snapshot: Snapshot;
  annotations: Annotation[];
}) {
  const integrated = snapshot.modes.integrated;
  const chat = snapshot.modes.chat;
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <p className={styles.eyebrow}>Eval scorecard</p>
        <h1>Integrated vs chat, scored against the same gold labels</h1>
        <p className={styles.lede}>
          {snapshot.n_items} test items, Sonnet 4.6 held constant across both
          modes (thinking on for integrated, off for chat). Last run{" "}
          {fmtTimestamp(snapshot.completed_at)} from commit{" "}
          <code className={styles.code}>{snapshot.git_sha.slice(0, 8)}</code>.
          Test set <code className={styles.code}>{snapshot.test_set_version}</code>.
        </p>
      </header>

      <section className={styles.section}>
        <h2>Headline</h2>
        <div className={styles.headlineGrid}>
          <HeadlineCard
            label="Classification accuracy"
            integrated={snapshot.headline.integrated.classification_accuracy}
            chat={snapshot.headline.chat.classification_accuracy}
            kind="percent"
            help="Per-item overall match across industry, segment, seniority, company_size."
          />
          <HeadlineCard
            label="Action accuracy"
            integrated={snapshot.headline.integrated.action_accuracy}
            chat={snapshot.headline.chat.action_accuracy}
            kind="percent"
            help="Does the model pick the gold action (auto_add/propose/discard/refuse)?"
          />
          <HeadlineCard
            label="Fit score, Spearman"
            integrated={snapshot.headline.integrated.fit_spearman}
            chat={snapshot.headline.chat.fit_spearman}
            kind="number"
            help="Spearman correlation between predicted and gold fit_score.value."
          />
          <HeadlineCard
            label="Substring grounding"
            integrated={snapshot.headline.integrated.substring_grounding_rate}
            chat={snapshot.headline.chat.substring_grounding_rate}
            kind="percent"
            help="Share of claims whose source_quote appears verbatim in the input."
          />
          <HeadlineCard
            label="Judge grounding (headline)"
            integrated={snapshot.headline.integrated.judge_grounding_rate}
            chat={snapshot.headline.chat.judge_grounding_rate}
            kind="percent"
            help="Lower of (Opus 4.7 grounding rate, GPT-5 grounding rate). Conservative read."
          />
          <HeadlineCard
            label="Hook pass rate"
            integrated={snapshot.headline.integrated.hook_pass_rate}
            chat={snapshot.headline.chat.hook_pass_rate}
            kind="percent"
            help="Single GPT-5-mini judge. Binary pass/fail with critique — no scales."
          />
          <HeadlineCard
            label="Latency p50"
            integrated={snapshot.headline.integrated.latency_p50_ms}
            chat={snapshot.headline.chat.latency_p50_ms}
            kind="ms"
            help="Per-call wall clock on the live eval pass (non-batch)."
            lowerIsBetter
          />
          <HeadlineCard
            label="Latency p95"
            integrated={snapshot.headline.integrated.latency_p95_ms}
            chat={snapshot.headline.chat.latency_p95_ms}
            kind="ms"
            help="Per-call wall clock, 95th percentile."
            lowerIsBetter
          />
        </div>
        {snapshot.headline.chat.extractor_complete_rate !== undefined && (
          <p className={styles.note}>
            Chat extractor complete:{" "}
            <strong>
              {fmtPercent(snapshot.headline.chat.extractor_complete_rate)}
            </strong>
            . Chat ran up to three user turns; the loop stops as soon as
            the Haiku 4.5 extractor reports every gold-shape field
            present. Average turns used:{" "}
            <strong>
              {fmtNumber(snapshot.headline.chat.avg_turns_used, 2)}
            </strong>
            . Cap-hit rate (3 turns without completion, scored as failure):{" "}
            <strong>
              {fmtPercent(snapshot.headline.chat.cap_hit_rate)}
            </strong>
            .
          </p>
        )}
      </section>

      {annotations.length > 0 && (
        <section className={styles.section}>
          <h2>Eval-and-fix loop</h2>
          <p className={styles.note}>
            Dated incidents where an eval-pass failure was diagnosed and a
            shipped fix moved the affected metric. Both pre-fix and post-fix
            snapshots stay in the repo so prospects can audit the loop, not
            only the latest number.
          </p>
          <ol className={styles.timeline}>
            {annotations.map((a, i) => (
              <li key={i} className={styles.timelineItem}>
                <div className={styles.timelineDate}>{a.date}</div>
                <div>
                  <div className={styles.timelineMetric}>
                    {a.metric}
                    {a.perturbation_class
                      ? ` · ${a.perturbation_class} perturbation`
                      : ""}
                  </div>
                  <p>
                    <strong>Failure:</strong> {a.failure_summary}
                  </p>
                  <p>
                    <strong>Fix:</strong> {a.fix_summary}
                  </p>
                  <p className={styles.timelineDelta}>
                    {fmtPercent(a.pre_fix_value)} → {fmtPercent(a.post_fix_value)}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}

      <section className={styles.section}>
        <h2>Per-mode breakdown</h2>
        <div className={styles.modeGrid}>
          <ModeColumn title="Integrated" mode={integrated} />
          <ModeColumn title="Chat" mode={chat} />
        </div>
      </section>

      {snapshot.robustness && (
        <section className={styles.section}>
          <h2>Robustness</h2>
          <p className={styles.note}>
            Three perturbation variants per base item: typos (per-word noise),
            sentence_reorder (neighbouring-sentence swaps), and an injection
            probe appended to the input. The reported drop is in
            classification accuracy and substring grounding rate vs. the
            main pass.
          </p>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Variant</th>
                <th>n</th>
                <th>Integrated classification</th>
                <th>Chat classification</th>
                <th>Integrated grounding</th>
                <th>Chat grounding</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(snapshot.robustness.by_variant).map(
                ([variant, v]) => (
                  <tr key={variant}>
                    <td>
                      <code className={styles.code}>{variant}</code>
                    </td>
                    <td>{v.n}</td>
                    <td>{fmtPercent(v.integrated.classification_accuracy)}</td>
                    <td>{fmtPercent(v.chat.classification_accuracy)}</td>
                    <td>
                      {fmtPercent(v.integrated.substring_grounded_rate)}
                    </td>
                    <td>{fmtPercent(v.chat.substring_grounded_rate)}</td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </section>
      )}

      <section className={styles.section}>
        <h2>By test-set kind</h2>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Kind</th>
              <th>n</th>
              <th>Integrated action</th>
              <th>Chat action</th>
              <th>Integrated classification</th>
              <th>Chat classification</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(snapshot.by_kind).map(([kind, v]) => (
              <tr key={kind}>
                <td>
                  <code className={styles.code}>{kind}</code>
                </td>
                <td>{v.n}</td>
                <td>{fmtPercent(v.integrated.action_accuracy)}</td>
                <td>{fmtPercent(v.chat.action_accuracy)}</td>
                <td>{fmtPercent(v.integrated.classification_accuracy)}</td>
                <td>{fmtPercent(v.chat.classification_accuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className={styles.section}>
        <h2>Failure modes</h2>
        <p className={styles.note}>
          Items where the predicted action, classification, or grounding
          differs from gold. Surfaced as concrete misses so we don&apos;t
          claim aggregate numbers without owning the failures behind them.
        </p>
        <div className={styles.modeGrid}>
          <FailureColumn
            title="Integrated misses"
            modes={integrated.failure_modes}
          />
          <FailureColumn title="Chat misses" modes={chat.failure_modes} />
        </div>
      </section>

      <footer className={styles.footer}>
        Models · integrated{" "}
        <code className={styles.code}>{snapshot.models.integrated}</code> · chat{" "}
        <code className={styles.code}>{snapshot.models.chat}</code> · extractor{" "}
        <code className={styles.code}>{snapshot.models.extractor}</code> ·
        grounding judges{" "}
        <code className={styles.code}>
          {snapshot.models.grounding_judges.anthropic}
        </code>{" "}
        +{" "}
        <code className={styles.code}>
          {snapshot.models.grounding_judges.openai}
        </code>{" "}
        · hook judge{" "}
        <code className={styles.code}>{snapshot.models.hook_judge}</code>
      </footer>
    </main>
  );
}

function HeadlineCard({
  label,
  integrated,
  chat,
  kind,
  help,
  lowerIsBetter = false,
}: {
  label: string;
  integrated: number | null;
  chat: number | null;
  kind: "percent" | "number" | "ms";
  help: string;
  lowerIsBetter?: boolean;
}) {
  const fmt =
    kind === "percent" ? fmtPercent : kind === "number" ? fmtNumber : fmtMs;
  const delta =
    integrated !== null && chat !== null ? integrated - chat : null;
  const sign = delta === null ? 0 : Math.sign(delta) * (lowerIsBetter ? -1 : 1);
  return (
    <article className={styles.headlineCard}>
      <header>
        <h3>{label}</h3>
        <p className={styles.help}>{help}</p>
      </header>
      <dl className={styles.headlineValues}>
        <div>
          <dt>Integrated</dt>
          <dd>{fmt(integrated)}</dd>
        </div>
        <div>
          <dt>Chat</dt>
          <dd>{fmt(chat)}</dd>
        </div>
      </dl>
      {delta !== null && (
        <p
          className={
            styles.delta +
            " " +
            (sign > 0
              ? styles.deltaUp
              : sign < 0
                ? styles.deltaDown
                : styles.deltaEq)
          }
        >
          Δ {kind === "percent" ? fmtPercent(delta) : fmt(delta)}
        </p>
      )}
    </article>
  );
}

function ModeColumn({ title, mode }: { title: string; mode: ModeBlock }) {
  const a = mode.aggregate;
  return (
    <div className={styles.modeColumn}>
      <h3>{title}</h3>
      <dl className={styles.kv}>
        <KV k="Success rate" v={fmtPercent(a.success_rate)} />
        <KV
          k="Classification — industry"
          v={fmtPercent(a.classification_per_field.industry)}
        />
        <KV
          k="Classification — segment"
          v={fmtPercent(a.classification_per_field.segment)}
        />
        <KV
          k="Classification — seniority"
          v={fmtPercent(a.classification_per_field.seniority)}
        />
        <KV
          k="Classification — company_size"
          v={fmtPercent(a.classification_per_field.company_size)}
        />
        <KV k="Fit Pearson" v={fmtNumber(a.fit_pearson)} />
        <KV k="Fit MAE" v={fmtNumber(a.fit_mae, 3)} />
        <KV
          k="Action accuracy"
          v={fmtPercent(a.action_accuracy)}
        />
        <KV
          k="Refuse-when-should"
          v={`${a.refuse_when_should_correct}/${a.refuse_when_should_total}`}
        />
        <KV
          k="Adversarial pass"
          v={
            a.adversarial_pass_rate !== null
              ? `${fmtPercent(a.adversarial_pass_rate)} (n=${a.adversarial_n})`
              : "—"
          }
        />
        <KV
          k="Substring grounding"
          v={fmtPercent(a.substring_grounded_rate)}
        />
        <KV
          k="Judge grounding (Opus)"
          v={fmtPercent(mode.grounding.opus_grounding_rate)}
        />
        <KV
          k="Judge grounding (OpenAI)"
          v={fmtPercent(mode.grounding.openai_grounding_rate)}
        />
        <KV
          k="Inter-judge kappa"
          v={fmtNumber(mode.grounding.cohen_kappa)}
        />
        <KV
          k="Hook pass rate"
          v={
            mode.hooks.pass_rate !== null
              ? `${fmtPercent(mode.hooks.pass_rate)} (n=${mode.hooks.n_scored})`
              : "—"
          }
        />
        <KV
          k="Tokens in p50 / p95"
          v={`${fmtNumber(a.tokens_in_p50, 0)} / ${fmtNumber(a.tokens_in_p95, 0)}`}
        />
        <KV
          k="Tokens out p50 / p95"
          v={`${fmtNumber(a.tokens_out_p50, 0)} / ${fmtNumber(a.tokens_out_p95, 0)}`}
        />
      </dl>
      <details className={styles.dims}>
        <summary>Per-dimension correlation</summary>
        <dl className={styles.kv}>
          {Object.entries(a.dim_correlations).map(([dim, r]) => (
            <KV
              key={dim}
              k={dim}
              v={`r = ${fmtNumber(r)} · MAE ${fmtNumber(a.dim_mae[dim], 3)}`}
            />
          ))}
        </dl>
      </details>
    </div>
  );
}

function FailureColumn({
  title,
  modes,
}: {
  title: string;
  modes: { item_id: string; kind: string; reasons: string[] }[];
}) {
  if (!modes.length) {
    return (
      <div className={styles.modeColumn}>
        <h3>{title}</h3>
        <p className={styles.note}>No misses on this pass.</p>
      </div>
    );
  }
  return (
    <div className={styles.modeColumn}>
      <h3>
        {title} <span className={styles.help}>({modes.length})</span>
      </h3>
      <ul className={styles.failures}>
        {modes.map((m) => (
          <li key={m.item_id}>
            <Link
              href={`/eval/item/${m.item_id}`}
              className={styles.failureLink}
            >
              <code className={styles.code}>
                {m.item_id} · {m.kind}
              </code>
            </Link>
            <ul>
              {m.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className={styles.kvRow}>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}

function fmtTimestamp(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toUTCString();
  } catch {
    return iso;
  }
}
