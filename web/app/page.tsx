import Link from "next/link";

export default function HomePage() {
  return (
    <main>
      <p
        style={{
          fontSize: "0.7rem",
          fontWeight: 700,
          letterSpacing: "0.15em",
          textTransform: "uppercase",
          color: "var(--amber)",
          margin: "0 0 0.35rem",
        }}
      >
        Live demo
      </p>
      <h1>Lead enrichment, built two ways</h1>
      <p style={{ color: "var(--muted)", maxWidth: "62ch" }}>
        Two realistic products, same model, same UI polish. The integrated
        build runs Sonnet 4.6 through a strict tool schema with extended
        thinking and a claim-grounding rule. The chat build runs the same
        model with a task-describing system prompt and no tools, no schema,
        no grounding rule. Paste a profile, watch both UIs handle it, then
        read the eval scorecard. Neither side is a strawman — the asymmetry
        sits at the model-side contract.
      </p>
      <ul
        style={{
          listStyle: "none",
          padding: 0,
          margin: "1.5rem 0",
          display: "grid",
          gap: "0.5rem",
        }}
      >
        <li>
          <Link href="/integrated">→ Integrated build</Link>
        </li>
        <li>
          <Link href="/chat">→ Chat build</Link>
        </li>
        <li>
          <Link href="/scorecard">→ Eval scorecard</Link>
        </li>
        <li>
          <Link href="/methodology">→ Methodology</Link>
        </li>
      </ul>
    </main>
  );
}
