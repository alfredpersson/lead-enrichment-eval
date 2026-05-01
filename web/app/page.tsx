import Link from "next/link";

export default function HomePage() {
  return (
    <main>
      <h1>Lead enrichment — integrated vs. chat</h1>
      <p>
        A B2B lead enrichment feature, built two ways. Same model, different
        output contracts. Paste a profile and a company, watch both UIs handle
        it, then read the eval scorecard.
      </p>
      <ul>
        <li>
          <Link href="/integrated">Integrated build →</Link>
        </li>
        <li>
          <Link href="/chat">Chat build →</Link>
        </li>
        <li>
          <Link href="/scorecard">Eval scorecard →</Link>
        </li>
        <li>
          <Link href="/methodology">Methodology →</Link>
        </li>
      </ul>
      <p style={{ opacity: 0.6, fontSize: "0.9rem" }}>
        Phase 1 scaffold. UI lands in Phase 2/3.
      </p>
    </main>
  );
}
