import Link from "next/link";
import { notFound } from "next/navigation";

import {
  fmtNumber,
  fmtPercent,
  loadSnapshot,
  loadTestItem,
  type EvalTestItem,
  type PerItemScore,
  type PredictedClaim,
  type PredictedOutput,
  type Snapshot,
} from "@/lib/scorecard";

import styles from "./item.module.css";

export const revalidate = 300;

export default async function EvalItemPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const item = loadTestItem(id);
  const snapshot = loadSnapshot();
  if (!item || !snapshot) {
    notFound();
  }

  const inputText = item.company
    ? `${item.profile}\n\n${item.company}`
    : item.profile;
  const integratedPred = snapshot.modes.integrated.raw_outputs?.[id] ?? null;
  const chatPred = snapshot.modes.chat.raw_outputs?.[id] ?? null;
  const integratedScore = findScore(snapshot, "integrated", id);
  const chatScore = findScore(snapshot, "chat", id);
  const integratedFailure = snapshot.modes.integrated.failure_modes.find(
    (f) => f.item_id === id,
  );
  const chatFailure = snapshot.modes.chat.failure_modes.find(
    (f) => f.item_id === id,
  );

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <p className={styles.eyebrow}>
          <Link href="/scorecard" className={styles.backLink}>
            ← Scorecard
          </Link>
        </p>
        <h1>
          Item {item.id} <span className={styles.kind}>· {item.kind}</span>
        </h1>
        {item.label && <p className={styles.label}>{item.label}</p>}
        {item.scenario && (
          <p className={styles.scenario}>
            scenario <code className={styles.code}>{item.scenario}</code>
          </p>
        )}
      </header>

      <section className={styles.section}>
        <h2>Input</h2>
        <pre className={styles.input}>{inputText}</pre>
      </section>

      <section className={styles.section}>
        <h2>Gold</h2>
        <GoldBlock item={item} />
      </section>

      <section className={styles.section}>
        <h2>Predictions</h2>
        <div className={styles.predGrid}>
          <PredictionColumn
            title="Integrated"
            failure={integratedFailure?.reasons ?? []}
            score={integratedScore}
            pred={integratedPred}
            inputText={inputText}
          />
          <PredictionColumn
            title="Chat"
            failure={chatFailure?.reasons ?? []}
            score={chatScore}
            pred={chatPred}
            inputText={inputText}
          />
        </div>
      </section>
    </main>
  );
}

function findScore(
  snapshot: Snapshot,
  mode: "integrated" | "chat",
  id: string,
): PerItemScore | null {
  return (
    snapshot.modes[mode].per_item.find((s) => s.item_id === id) ?? null
  );
}

function GoldBlock({ item }: { item: EvalTestItem }) {
  const cls = item.gold.classification ?? {};
  const fit = item.gold.fit_score;
  return (
    <dl className={styles.kv}>
      {item.gold.expected_action && (
        <KV k="Expected action" v={item.gold.expected_action} />
      )}
      {fit && <KV k="Fit score" v={fit.value.toFixed(2)} />}
      <KV k="Industry" v={cls.industry ?? "—"} />
      <KV k="Segment" v={cls.segment ?? "—"} />
      <KV k="Seniority" v={cls.seniority ?? "—"} />
      <KV k="Company size" v={cls.company_size ?? "—"} />
      {item.gold.input_lang && <KV k="Language" v={item.gold.input_lang} />}
      {item.gold.notes && (
        <div className={styles.kvNotes}>
          <dt>Notes</dt>
          <dd>{item.gold.notes}</dd>
        </div>
      )}
      {item.gold.adversarial_pass_criteria &&
        item.gold.adversarial_pass_criteria.length > 0 && (
          <div className={styles.kvNotes}>
            <dt>Adversarial pass criteria</dt>
            <dd>
              <ul className={styles.criteria}>
                {item.gold.adversarial_pass_criteria.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </dd>
          </div>
        )}
    </dl>
  );
}

function PredictionColumn({
  title,
  failure,
  score,
  pred,
  inputText,
}: {
  title: string;
  failure: string[];
  score: PerItemScore | null;
  pred: PredictedOutput | null;
  inputText: string;
}) {
  if (!pred) {
    return (
      <div className={styles.predColumn}>
        <h3>{title}</h3>
        <p className={styles.muted}>No output recorded for this item.</p>
      </div>
    );
  }
  const action = pred.action ?? "—";
  const fit = pred.fit_score;
  const cls = pred.classification ?? {};
  return (
    <div className={styles.predColumn}>
      <h3>
        {title}
        {failure.length > 0 ? (
          <span className={styles.failBadge}>miss</span>
        ) : (
          <span className={styles.passBadge}>pass</span>
        )}
      </h3>
      {failure.length > 0 && (
        <ul className={styles.failures}>
          {failure.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      )}
      <dl className={styles.kv}>
        <KV
          k="Action"
          v={
            score && score.action_gold
              ? `${action} ${score.action_correct ? "✓" : `✗ (gold: ${score.action_gold})`}`
              : action
          }
        />
        {fit && (
          <KV
            k="Fit"
            v={
              score && score.fit_value_gold !== null
                ? `${fit.value.toFixed(2)} (gold ${score.fit_value_gold.toFixed(2)}, |Δ| ${fmtNumber(score.fit_value_abs_error, 2)})`
                : fit.value.toFixed(2)
            }
          />
        )}
        <KV
          k="Industry"
          v={`${cls.industry ?? "—"}${matchMark(score?.classification_match.industry)}`}
        />
        <KV
          k="Segment"
          v={`${cls.segment ?? "—"}${matchMark(score?.classification_match.segment)}`}
        />
        <KV
          k="Seniority"
          v={`${cls.seniority ?? "—"}${matchMark(score?.classification_match.seniority)}`}
        />
        <KV
          k="Company size"
          v={`${cls.company_size ?? "—"}${matchMark(score?.classification_match.company_size)}`}
        />
        {score && (
          <KV
            k="Grounding"
            v={
              score.claim_count > 0
                ? `${score.substring_grounded_count}/${score.claim_count} claims grounded (${fmtPercent(
                    score.substring_grounded_rate,
                  )})`
                : "no claims"
            }
          />
        )}
      </dl>
      {pred.draft_hook?.text && (
        <details className={styles.hook}>
          <summary>Draft hook</summary>
          <p>{pred.draft_hook.text}</p>
          {pred.draft_hook.rationale && (
            <p className={styles.muted}>{pred.draft_hook.rationale}</p>
          )}
        </details>
      )}
      {pred.claims && pred.claims.length > 0 && (
        <details className={styles.claims}>
          <summary>Claims ({pred.claims.length})</summary>
          <ol>
            {pred.claims.map((c, i) => (
              <ClaimRow key={i} claim={c} inputText={inputText} />
            ))}
          </ol>
        </details>
      )}
      {pred.reasoning && (
        <details className={styles.reasoning}>
          <summary>Reasoning</summary>
          <p>{pred.reasoning}</p>
        </details>
      )}
    </div>
  );
}

function ClaimRow({
  claim,
  inputText,
}: {
  claim: PredictedClaim;
  inputText: string;
}) {
  const grounded = claim.source_quote
    ? inputText.includes(claim.source_quote)
    : false;
  return (
    <li>
      <div>{claim.text}</div>
      <div className={styles.quote}>
        <span className={grounded ? styles.quoteOk : styles.quoteBad}>
          {grounded ? "✓ in input" : "✗ not in input"}
        </span>{" "}
        <code className={styles.code}>{claim.source_quote || "(empty)"}</code>
      </div>
    </li>
  );
}

function matchMark(matched: boolean | undefined): string {
  if (matched === undefined) return "";
  return matched ? " ✓" : " ✗";
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className={styles.kvRow}>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}
