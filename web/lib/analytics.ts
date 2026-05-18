import { track as vercelTrack } from "@vercel/analytics";

// Operational events (Plan §Observability layer 1) and prospect-funnel events
// (layer 2). Methodology §11 names the funnel signals explicitly so prospects
// know what's instrumented and what isn't — keep these in sync with that page.
type AnalyticsEvent =
  | { name: "example-loaded"; props: { surface: "integrated" | "chat"; exampleId: string } }
  | { name: "mode-switched"; props: { to: "integrated" | "chat" } }
  | { name: "completion"; props: { surface: "integrated" | "chat"; action?: string; fitScore?: number } }
  | { name: "error"; props: { surface: string; kind: string } }
  | { name: "scorecard-viewed"; props?: Record<string, never> }
  | { name: "methodology-viewed"; props?: Record<string, never> }
  | { name: "own-input-pasted"; props: { surface: "integrated" | "chat" } }
  | { name: "scorecard-clicked-from-result"; props: { surface: "integrated" | "chat" } }
  | { name: "page-bounce-after-low-fit"; props: { fitScore: number } }
  // Defined so the methodology page §11 listing stays honest. Not yet wired —
  // there's no in-product contact CTA on /integrated or /chat. Wire when one
  // ships (likely on the result drawer's empty/low-fit state, or a "Run on
  // your pipeline" CTA tied to the v2 hook).
  | { name: "contact-link-clicked-from-result"; props: { surface: "integrated" | "chat" } };

export function track<E extends AnalyticsEvent>(event: E): void {
  const props = (event as { props?: Record<string, string | number | boolean | null> }).props;
  vercelTrack(event.name, props);
}
