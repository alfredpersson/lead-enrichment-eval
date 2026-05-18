// Types and loader for the eval JSON snapshot. The snapshot is produced by
// `services.eval.runner` and committed to `data/eval_runs/latest.json`. The
// scorecard server-component reads it via `fs` at request time.

import fs from "node:fs";
import path from "node:path";

export interface AggregateMetrics {
  n: number;
  success_rate: number;
  classification_accuracy: number;
  classification_per_field: Record<string, number>;
  action_accuracy: number;
  fit_pearson: number | null;
  fit_spearman: number | null;
  fit_mae: number | null;
  dim_correlations: Record<string, number | null>;
  dim_mae: Record<string, number | null>;
  substring_grounded_rate: number;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  tokens_in_p50: number | null;
  tokens_in_p95: number | null;
  tokens_out_p50: number | null;
  tokens_out_p95: number | null;
  adversarial_pass_rate: number | null;
  adversarial_n: number;
  refuse_when_should_correct: number;
  refuse_when_should_total: number;
}

export interface PerItemScore {
  item_id: string;
  kind: string;
  success: boolean;
  classification_match: Record<string, boolean>;
  classification_overall: boolean;
  fit_value_predicted: number | null;
  fit_value_gold: number | null;
  fit_value_abs_error: number | null;
  fit_dimensions_predicted: Record<string, number>;
  fit_dimensions_gold: Record<string, number>;
  fit_dimensions_abs_error: Record<string, number>;
  action_predicted: string | null;
  action_gold: string | null;
  action_correct: boolean;
  claim_count: number;
  substring_grounded_count: number;
  substring_grounded_rate: number;
  adversarial_pass: boolean | null;
  adversarial_failures: string[];
}

export interface FailureMode {
  item_id: string;
  kind: string;
  reasons: string[];
}

export interface GroundingResults {
  n_claims: number;
  n_judged_by_opus: number;
  opus_grounding_rate: number | null;
  openai_grounding_rate: number | null;
  headline_grounding_rate: number | null;
  cohen_kappa: number | null;
  judgements: unknown[];
}

export interface HookResults {
  n_scored: number;
  pass_rate: number | null;
  judgements: unknown[];
}

export interface ModeBlock {
  aggregate: AggregateMetrics;
  grounding: GroundingResults;
  hooks: HookResults;
  per_item: PerItemScore[];
  raw_outputs?: Record<string, PredictedOutput>;
  inference_meta: Array<{
    item_id: string;
    success: boolean;
    error: string | null;
    latency_ms: number;
    input_tokens: number;
    output_tokens: number;
    thinking_tokens: number | null;
    cache_read_tokens: number;
    stop_reason: string | null;
  }>;
  extractor?: Array<{
    item_id: string;
    success: boolean;
    extractor_complete: boolean;
    error: string | null;
    input_tokens: number;
    output_tokens: number;
  }>;
  failure_modes: FailureMode[];
}

export interface RobustnessBlock {
  by_variant: Record<
    string,
    {
      n: number;
      integrated: AggregateMetrics;
      chat: AggregateMetrics;
      per_item_integrated: PerItemScore[];
      per_item_chat: PerItemScore[];
    }
  >;
  n_base_items: number;
}

export interface Headline {
  classification_accuracy: number;
  action_accuracy: number;
  fit_spearman: number | null;
  substring_grounding_rate: number;
  judge_grounding_rate: number | null;
  hook_pass_rate: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  extractor_complete_rate?: number;
  avg_turns_used?: number | null;
  cap_hit_rate?: number;
}

export interface Snapshot {
  schema_version: number;
  run_tag: string | null;
  git_sha: string;
  test_set_version: string;
  n_items: number;
  started_at: string;
  completed_at: string;
  elapsed_seconds: number;
  models: {
    integrated: string;
    chat: string;
    extractor: string;
    grounding_judges: { anthropic: string; openai: string };
    hook_judge: string;
  };
  headline: {
    integrated: Headline;
    chat: Headline;
  };
  modes: {
    integrated: ModeBlock;
    chat: ModeBlock;
  };
  by_kind: Record<
    string,
    { n: number; integrated: AggregateMetrics; chat: AggregateMetrics }
  >;
  robustness: RobustnessBlock | null;
}

export interface Annotation {
  date: string;
  metric: string;
  perturbation_class: string | null;
  failure_summary: string;
  fix_summary: string;
  pre_fix_snapshot: string;
  post_fix_snapshot: string;
  pre_fix_value: number;
  post_fix_value: number;
}

export interface AnnotationsFile {
  schema_version: number;
  annotations: Annotation[];
}

export interface EvalItemGold {
  input_lang?: string;
  structured_fields?: Record<string, unknown>;
  classification?: {
    industry?: string;
    segment?: string;
    seniority?: string;
    company_size?: string;
  };
  fit_score?: {
    value: number;
    dimensions?: Record<string, number>;
  };
  expected_action?: string;
  claims_allowed?: unknown;
  notes?: string;
  adversarial_pass_criteria?: string[];
}

export interface EvalTestItem {
  id: string;
  kind: string;
  scenario?: string;
  label?: string;
  profile: string;
  company?: string | null;
  gold: EvalItemGold;
}

export interface ItemsFile {
  version: string;
  items: EvalTestItem[];
}

export interface PredictedClaim {
  text: string;
  source_quote: string;
  confidence?: number;
}

export interface PredictedOutput {
  classification?: {
    industry?: string;
    segment?: string;
    seniority?: string;
    company_size?: string;
  };
  fit_score?: {
    value: number;
    dimensions?: Record<string, number>;
  };
  claims?: PredictedClaim[];
  draft_hook?: { text?: string; rationale?: string };
  action?: string;
  reasoning?: string;
}

// Files are mirrored from data/eval_runs/ into web/public/eval/ by the npm
// predev/prebuild hooks (see web/scripts/sync-eval-snapshot.mjs). Reading
// from public/ keeps the loader inside the Next.js project root so Vercel
// bundles the data with the deployment.
const EVAL_DIR = path.resolve(process.cwd(), "public", "eval");

export function loadSnapshot(): Snapshot | null {
  const p = path.join(EVAL_DIR, "latest.json");
  try {
    const raw = fs.readFileSync(p, "utf8");
    if (!raw.trim() || raw.trim() === "null") return null;
    return JSON.parse(raw) as Snapshot;
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw e;
  }
}

export function loadTestItems(): EvalTestItem[] {
  const p = path.join(EVAL_DIR, "items.json");
  try {
    const raw = fs.readFileSync(p, "utf8");
    const file = JSON.parse(raw) as ItemsFile;
    return file.items ?? [];
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw e;
  }
}

export function loadTestItem(id: string): EvalTestItem | null {
  return loadTestItems().find((it) => it.id === id) ?? null;
}

export function loadAnnotations(): Annotation[] {
  const p = path.join(EVAL_DIR, "annotations.json");
  try {
    const raw = fs.readFileSync(p, "utf8");
    const file = JSON.parse(raw) as AnnotationsFile;
    return file.annotations ?? [];
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw e;
  }
}

export function fmtPercent(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtNumber(
  v: number | null | undefined,
  digits = 2,
): string {
  if (v === null || v === undefined) return "—";
  return v.toFixed(digits);
}

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return Math.round(v).toLocaleString();
}

export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  if (v >= 1000) return `${(v / 1000).toFixed(2)}s`;
  return `${Math.round(v)}ms`;
}
