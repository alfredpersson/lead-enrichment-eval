// Copy the canonical eval snapshot + annotations from data/eval_runs/ into
// web/public/eval/ so they ship with the Vercel build. The scorecard reads
// from /eval/latest.json at request time. Run automatically via the npm
// predev/prebuild hooks so devs never see stale data in dev.
//
// If the canonical files don't exist yet (first run, no eval committed),
// write empty placeholders so the scorecard renders its empty state instead
// of 404'ing at build time.

import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..");
const src = resolve(repoRoot, "data", "eval_runs");
const dst = resolve(repoRoot, "web", "public", "eval");

mkdirSync(dst, { recursive: true });

function copyOrPlaceholder(name, placeholder) {
  const from = resolve(src, name);
  const to = resolve(dst, name);
  if (existsSync(from)) {
    writeFileSync(to, readFileSync(from));
    console.log(`[sync-eval-snapshot] ${name} ← ${from}`);
  } else {
    writeFileSync(to, JSON.stringify(placeholder, null, 2));
    console.log(
      `[sync-eval-snapshot] ${name} written as placeholder (source absent)`
    );
  }
}

copyOrPlaceholder("latest.json", null);
copyOrPlaceholder("annotations.json", {
  schema_version: 1,
  annotations: [],
});
