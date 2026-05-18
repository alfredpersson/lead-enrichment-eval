export type Seniority =
  | "IC"
  | "Manager"
  | "Director"
  | "VP"
  | "C-level"
  | "Founder";

export type CompanySize =
  | "1-10"
  | "11-50"
  | "51-200"
  | "201-500"
  | "500+";

export type Action = "auto_add" | "propose" | "discard" | "refuse";

export interface Classification {
  industry: string;
  segment: string;
  seniority: Seniority;
  company_size: CompanySize;
}

export interface FitDimensions {
  stage_match: number;
  headcount_match: number;
  arr_match: number;
  product_shape_match: number;
  role_match: number;
}

export interface FitScore {
  value: number;
  dimensions: FitDimensions;
}

export interface Claim {
  text: string;
  source_quote: string;
  confidence: number;
}

export interface DraftHook {
  text: string;
  claims_used: string[];
  confidence: number;
}

export interface EvalNeighbour {
  id: string;
  similarity: number;
  score: number | null;
}

export interface EnrichMeta {
  request_id: string;
  latency_ms: number;
  tokens_in: number;
  tokens_out: number;
  thinking_tokens: number | null;
  thinking_budget: number;
  cache_hit: boolean;
  model: string;
  eval_neighbours: EvalNeighbour[];
}

export interface EnrichOutput {
  classification: Classification;
  fit_score: FitScore;
  claims: Claim[];
  draft_hook: DraftHook;
  action: Action;
  reasoning: string;
  meta: EnrichMeta;
}

export type StreamEvent =
  | { type: "thinking"; delta: string }
  | { type: "result"; output: EnrichOutput }
  | { type: "error"; code: string; message: string };

export interface ConfirmedAction {
  ts: number;
  example_id: string | null;
  action: Action;
  fit: number;
  label: string;
}
