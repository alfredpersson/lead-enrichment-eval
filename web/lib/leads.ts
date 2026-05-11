// Lead-row state for the chat dashboard. The queue on /chat surfaces these
// rows; status pills survive reload via localStorage. /integrated uses its
// own in-memory row state since it carries streaming/scoring metadata that
// doesn't make sense to persist.

import { EXEMPLARS, type Exemplar } from "@/lib/exemplars";

export type LeadStatus = "new" | "working" | "auto_add" | "discard";

export interface LeadRow {
  id: string;
  source: "exemplar" | "paste";
  leadName: string;
  title: string;
  companyName: string;
  profile: string;
  company: string | null;
  status: LeadStatus;
  createdAt: number;
}

const STORAGE_KEY = "lead-enrichment.leads.v1";

function isBrowser(): boolean {
  return (
    typeof window !== "undefined" && typeof window.localStorage !== "undefined"
  );
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

export function deriveCompanyName(
  title: string,
  company: string | null,
): string {
  const match = title.match(/\s+(?:at|på|hos|bei|chez)\s+(.+)$/i);
  if (match) return match[1].trim();
  if (company) {
    const sentence = company.split(/[.!?]/)[0]?.trim() ?? "";
    const head = sentence.split(/\s+is\s+|\s+är\s+/i)[0]?.trim();
    if (head) return head;
  }
  return "";
}

export function exemplarToLead(ex: Exemplar): LeadRow {
  const name = firstLine(ex.profile) || ex.label;
  const title = nthNonEmptyLine(ex.profile, 1) || "";
  return {
    id: ex.id,
    source: "exemplar",
    leadName: name,
    title,
    companyName: deriveCompanyName(title, ex.company),
    profile: ex.profile,
    company: ex.company,
    status: "new",
    createdAt: Date.now(),
  };
}

export function buildPasteLead(
  id: string,
  profile: string,
  company: string | null,
): LeadRow {
  const name = firstLine(profile) || `Paste ${id}`;
  const title = nthNonEmptyLine(profile, 1) || "";
  return {
    id,
    source: "paste",
    leadName: name.slice(0, 80),
    title: title.slice(0, 120),
    companyName: deriveCompanyName(title, company),
    profile,
    company,
    status: "new",
    createdAt: Date.now(),
  };
}

function seed(): LeadRow[] {
  return EXEMPLARS.map(exemplarToLead);
}

export function loadLeads(): LeadRow[] {
  if (!isBrowser()) return seed();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return seed();
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return seed();
    return parsed.filter(isLeadRow);
  } catch {
    return seed();
  }
}

export function saveLeads(leads: LeadRow[]): void {
  if (!isBrowser()) return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(leads));
  } catch {
    // Quota: drop oldest paste leads and retry once.
    const pruned = leads.filter(
      (l) => l.source === "exemplar" || Date.now() - l.createdAt < 86_400_000,
    );
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(pruned));
    } catch {
      // give up
    }
  }
}

export const STATUS_LABELS: Record<LeadStatus, string> = {
  new: "New",
  working: "Working",
  auto_add: "Auto-add",
  discard: "Discard",
};

export const STATUS_ORDER: LeadStatus[] = [
  "new",
  "working",
  "auto_add",
  "discard",
];

function isLeadRow(value: unknown): value is LeadRow {
  if (!value || typeof value !== "object") return false;
  const r = value as Record<string, unknown>;
  return (
    typeof r.id === "string" &&
    typeof r.profile === "string" &&
    (r.company === null || typeof r.company === "string") &&
    typeof r.status === "string" &&
    typeof r.leadName === "string"
  );
}
