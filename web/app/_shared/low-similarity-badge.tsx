"use client";

import type { EvalNeighbour } from "@/lib/types";

const LOW_SIMILARITY_THRESHOLD = 0.5;

interface Props {
  neighbours: EvalNeighbour[];
  className: string;
}

export function LowSimilarityBadge({ neighbours, className }: Props) {
  if (neighbours.length === 0) return null;
  const max = neighbours.reduce((acc, n) => Math.max(acc, n.similarity), 0);
  if (max >= LOW_SIMILARITY_THRESHOLD) return null;
  return (
    <span className={className} role="note">
      low-similarity match: input is novel relative to the test set
    </span>
  );
}
