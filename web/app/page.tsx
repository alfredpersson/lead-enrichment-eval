import Link from "next/link";
import styles from "./home.module.css";

export default function HomePage() {
  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <p className={styles.eyebrow}>Live demo</p>
        <h1 className={styles.heroTitle}>
          Lead enrichment, built two ways
        </h1>
        <p className={styles.heroLede}>
          Paste a LinkedIn-style profile and a company description.
          The app scores fit against a fixed B2B SaaS ICP and
          returns source-quoted claims, an outreach hook, and a
          routing decision — auto-add, propose, or discard.
        </p>
        <p className={styles.heroLede}>
          Two builds run the same workflow against the same eval
          set. The integrated build uses a strict tool schema,
          extended thinking, and a per-claim grounding rule. The
          chat build uses a task-describing system prompt and
          nothing else. The scorecard measures the architectural
          gap.
        </p>
        <div className={styles.heroCtas}>
          <Link href="/integrated" className={styles.heroPrimary}>
            Try the integrated build →
          </Link>
          <Link href="/chat" className={styles.heroSecondary}>
            Try the chat build →
          </Link>
        </div>
      </section>

      <section className={styles.diff}>
        <p className={styles.eyebrow}>What&rsquo;s different</p>
        <div className={styles.diffTable}>
          <div className={`${styles.diffRow} ${styles.diffHead}`}>
            <div className={`${styles.diffCell} ${styles.diffHeadLabel}`}>
              Dimension
            </div>
            <div className={`${styles.diffCell} ${styles.diffHeadLabel}`}>
              Integrated
            </div>
            <div className={`${styles.diffCell} ${styles.diffHeadLabel}`}>
              Chat
            </div>
          </div>
          <DiffRow
            label="Output contract"
            integrated="Strict enrich_lead tool schema"
            chat="Free-form prose"
          />
          <DiffRow
            label="Grounding rule"
            integrated="Per-claim verbatim source quote"
            chat="No structural requirement"
          />
          <DiffRow
            label="Inference shape"
            integrated="Single structured call"
            chat="Multi-turn chat + Haiku extractor"
          />
          <DiffRow
            label="Extended thinking"
            integrated="4000-token budget"
            chat="Off"
          />
        </div>
      </section>

      <section>
        <p className={styles.eyebrow}>Explore</p>
        <ul className={styles.cardGrid}>
          <li>
            <DestinationCard
              href="/integrated"
              eyebrow="01"
              title="Integrated build"
              body="Lead queue with structured outputs, source-quoted claims, extended thinking deltas, and an under-the-hood telemetry drawer."
              cta="Open the queue"
            />
          </li>
          <li>
            <DestinationCard
              href="/chat"
              eyebrow="02"
              title="Chat build"
              body="Productized chat against the same model. No tools, no schema. Conversations stay in the browser; diagnostics behind a toggle."
              cta="Open the chat"
            />
          </li>
          <li>
            <DestinationCard
              href="/scorecard"
              eyebrow="03"
              title="Eval scorecard"
              body="73-item test set, both modes scored against the same gold labels, two grounding judges, and a dated pre/post-fix snapshot."
              cta="See the numbers"
            />
          </li>
          <li>
            <DestinationCard
              href="/methodology"
              eyebrow="04"
              title="Methodology"
              body="What's measured, how the labels were built, the criteria revision log, judge models, and the known limits to read before the numbers."
              cta="Read the methodology"
            />
          </li>
        </ul>
      </section>
    </main>
  );
}

function DiffRow({
  label,
  integrated,
  chat,
}: {
  label: string;
  integrated: string;
  chat: string;
}) {
  return (
    <div className={styles.diffRow}>
      <div className={`${styles.diffCell} ${styles.diffLabel}`}>{label}</div>
      <div className={`${styles.diffCell} ${styles.diffIntegrated}`}>
        {integrated}
      </div>
      <div className={`${styles.diffCell} ${styles.diffChat}`}>{chat}</div>
    </div>
  );
}

function DestinationCard({
  href,
  eyebrow,
  title,
  body,
  cta,
}: {
  href: string;
  eyebrow: string;
  title: string;
  body: string;
  cta: string;
}) {
  return (
    <Link href={href} className={styles.card}>
      <span className={styles.cardEyebrow}>{eyebrow}</span>
      <h2 className={styles.cardTitle}>{title}</h2>
      <p className={styles.cardBody}>{body}</p>
      <span className={styles.cardCta}>{cta} →</span>
    </Link>
  );
}
