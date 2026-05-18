"use client";

import { useEffect } from "react";

import { track } from "./analytics";

// Fires a methodology-viewed or scorecard-viewed custom event when the page
// mounts. Vercel Analytics already auto-tracks pageviews; this is the named
// version that makes funnel analysis easier (see methodology §11).
export function PageViewTracker({ name }: { name: "scorecard-viewed" | "methodology-viewed" }) {
  useEffect(() => {
    track({ name });
  }, [name]);
  return null;
}
