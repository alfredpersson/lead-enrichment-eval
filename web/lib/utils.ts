export function isBrowser(): boolean {
  return (
    typeof window !== "undefined" && typeof window.localStorage !== "undefined"
  );
}

export function firstLine(text: string): string {
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed) return trimmed;
  }
  return "";
}

export function nthNonEmptyLine(text: string, n: number): string {
  let i = 0;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (i === n) return trimmed;
    i += 1;
  }
  return "";
}

export function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}
