import styles from "./methodology.module.css";

export const metadata = {
  title: "Methodology · Lead Enrichment",
};

export default function MethodologyPage() {
  return (
    <main className={styles.page}>
      <header>
        <p className={styles.eyebrow}>Methodology</p>
        <h1>How the eval scorecard is built</h1>
        <p className={styles.lede}>
          What is being measured, the gold labels behind it, the judges, and
          the known limits. Read this before reading numbers.
        </p>
      </header>

      <nav className={styles.toc} aria-label="contents">
        <strong>Contents</strong>
        <ol>
          <li>
            <a href="#thesis">Thesis and what is being measured</a>
          </li>
          <li>
            <a href="#icp">The fictional ICP</a>
          </li>
          <li>
            <a href="#test-set">Test set composition</a>
          </li>
          <li>
            <a href="#synthesis">Synthesis pipeline</a>
          </li>
          <li>
            <a href="#criteria-revision">Criteria revision log</a>
          </li>
          <li>
            <a href="#deterministic">Deterministic metrics</a>
          </li>
          <li>
            <a href="#judges">LLM-as-judge metrics</a>
          </li>
          <li>
            <a href="#robustness">Robustness methodology</a>
          </li>
          <li>
            <a href="#cross-mode">Cross-mode comparison</a>
          </li>
          <li>
            <a href="#ops">Operations</a>
          </li>
          <li>
            <a href="#telemetry">Telemetry vs adoption</a>
          </li>
          <li>
            <a href="#production">Input shape in production</a>
          </li>
          <li>
            <a href="#limits">Known limits</a>
          </li>
          <li>
            <a href="#custom-icp">Custom ICP, v2</a>
          </li>
        </ol>
      </nav>

      <section id="thesis" className={styles.section}>
        <h2>1. Thesis and what is being measured</h2>
        <p>
          Two realistic products, the same Sonnet 4.6 model, the same UI
          polish. The integrated build runs the model through a strict{" "}
          <code className={styles.code}>enrich_lead</code> tool with a JSON
          schema, extended thinking, and a claim-grounding rule that requires
          every claim to cite a verbatim source quote. The chat build runs
          the same model with a task-describing system prompt, with no
          tools, no schema, no grounding rule. The eval scorecard measures how that
          architectural asymmetry surfaces on the same gold labels.
        </p>
        <p>
          Each test item carries the inputs a salesperson would see (a
          profile, optionally a company description) plus gold labels for
          structured fields, classification, a 0.0–1.0 fit score with five
          named dimensions, the claims a hook may use with verbatim source
          quotes, and the expected action.
        </p>
      </section>

      <section id="icp" className={styles.section}>
        <h2>2. The fictional ICP</h2>
        <p>
          The demo ships with a fixed ICP. It is declared explicitly so
          prospects can see what is being scored.
        </p>
        <ul>
          <li>Stages: Series A, Series B, Series C.</li>
          <li>Headcount: 20 to 250.</li>
          <li>ARR: $2M to $50M USD.</li>
          <li>
            Product shape: B2B SaaS company shipping at least one user-facing
            AI feature, or with one in active development.
          </li>
          <li>
            Target roles: VP Product, Head of AI/ML, Director of Engineering,
            technical Founder/CTO.
          </li>
        </ul>
        <h3>Action thresholds</h3>
        <p>
          Fit score is on a 0.0–1.0 scale across the entire demo. Evaluation
          order is top to bottom. The first matching row wins.
        </p>
        <ol>
          <li>
            <code className={styles.code}>refuse</code> when the input lacks
            data to judge.
          </li>
          <li>
            <code className={styles.code}>propose</code> when any claim is
            ungrounded, regardless of fit score.
          </li>
          <li>
            <code className={styles.code}>auto_add</code> when fit &gt; 0.80
            and every claim grounded.
          </li>
          <li>
            <code className={styles.code}>propose</code> when fit is in [0.50,
            0.80] and every claim grounded.
          </li>
          <li>
            <code className={styles.code}>discard</code> when fit &lt; 0.50
            and every claim grounded.
          </li>
        </ol>
        <p className={styles.callout}>
          <code className={styles.code}>refuse</code> and{" "}
          <code className={styles.code}>discard</code> are different.{" "}
          <em>Refuse</em> means &ldquo;cannot judge.&rdquo; <em>Discard</em>{" "}
          means &ldquo;judged and rejected.&rdquo;
        </p>
      </section>

      <section id="test-set" className={styles.section}>
        <h2>3. Test set composition</h2>
        <p>
          73 items in v1.0. Languages covered: English, Swedish, German.
          Inputs in other languages return an out-of-scope response without
          burning a model call.
        </p>
        <ul>
          <li>
            <strong>5 hand-crafted exemplars</strong> chosen to span clean
            fit, ambiguous fit, sparse input, adversarial input (with a
            prompt injection in the bio), and a multilingual case. These
            double as the example cards in both UIs and as conversation
            starters in the chat build.
          </li>
          <li>
            <strong>36 fully synthetic profiles</strong> shaped like LinkedIn
            entries and B2B SaaS company descriptions. Hand-labelled. Roughly
            10% omit the company description to cover the company-optional
            path.
          </li>
          <li>
            <strong>20 adversarial cases:</strong> 6 prompt injections in
            profile or company text, 4 jailbreak / instruction-override
            attempts, 4 contradictory data (claims vs job title mismatch), 3
            sparse input (one or two fields only), 3 multilingual injection.
          </li>
          <li>
            <strong>12 edge cases:</strong> very short, very long, ambiguous
            job titles, supported non-English.
          </li>
        </ul>
        <p>
          Test-set versioning: every eval run is tagged with the test set&apos;s
          git SHA and version label (currently v1.0). Trends are computed
          within a version. Material test-set changes bump the label and the
          methodology page lists the change.
        </p>
      </section>

      <section id="synthesis" className={styles.section}>
        <h2>4. Synthesis pipeline</h2>
        <p>
          Synthetic profiles are LLM-generated from short scenario specs
          (e.g.{" "}
          <em>
            &ldquo;Series B HR-tech VP Product, attrition feature shipped&rdquo;
          </em>
          ), then hand-reviewed and hand-labelled. No real LinkedIn data, no
          scraping. The generation prompts and scenario specs live in the
          repo at <code className={styles.code}>scripts/generate_*.py</code>{" "}
          and <code className={styles.code}>data/*_specs.jsonl</code>.
        </p>
        <p>
          v1 was solo-labelled, which is the largest open methodological
          risk and is named below under known limits. v1.1 will add a 20%
          peer second-pass on fit-score and claims-allowed labels with
          Cohen&apos;s kappa reported on the scorecard.
        </p>
      </section>

      <section id="criteria-revision" className={styles.section}>
        <h2>5. Criteria revision log</h2>
        <p>
          Shreya Shankar&apos;s{" "}
          <em>
            <a
              href="https://arxiv.org/abs/2404.12272"
              target="_blank"
              rel="noreferrer"
            >
              Who Validates the Validators?
            </a>
          </em>{" "}
          (UIST 2024) names <strong>criteria drift</strong>: some eval
          criteria are output-dependent and cannot be fully specified up
          front. Pretending otherwise produces an evaluation that grades
          against an unrealistic rubric. This section logs every rubric
          anchor that was edited after seeing model or labeller output, so
          the cost of those edits is auditable instead of buried in a git
          history.
        </p>
        <dl>
          <dt>
            <strong>2026-05-08 ·</strong>{" "}
            <code className={styles.code}>product_shape_match</code> ·
            new 0.75 anchor: &ldquo;B2B SaaS surface present but secondary
            to consumer revenue&rdquo;
          </dt>
          <dd>
            The gap surfaced while labelling Exemplar 2 (Moodboard, a Series A
            consumer-led app with a B2B API side). The original rubric
            treated B2B SaaS as binary, so a hybrid landed on 1.0 by
            virtue of the AI-feature signal even though the B2B surface
            was secondary to consumer revenue. The new OR-condition lets
            hybrids land on 0.75. Exemplar 2&apos;s dimension moved from
            1.00 to 0.75; applied retroactively before bulk labelling
            started.
          </dd>
          <dt>
            <strong>2026-05-08 ·</strong>{" "}
            <code className={styles.code}>role_match</code> · new 0.75
            anchor: &ldquo;Founder/CEO at target-shape company without
            confirmed technical background&rdquo;
          </dt>
          <dd>
            The same rubric review surfaced this anchor. The original
            rubric reserved 1.0 for technical Founder/CTO and dropped
            non-technical founders to 0.25 via
            &ldquo;mismatched on either axis.&rdquo; That ignored
            decision authority, which is the core thing{" "}
            <code className={styles.code}>target_roles</code> is meant to
            capture. The new OR-condition recognises non-technical
            founder authority as a partial match. Exemplar 2&apos;s
            dimension moved from 0.25 to 0.75.
          </dd>
          <dt>
            <strong>2026-05-08 ·</strong> Exemplar 2 holistic{" "}
            <code className={styles.code}>fit_score.value</code> · 0.55
            → 0.65
          </dt>
          <dd>
            This entry re-justifies the holistic fit score without
            editing the rubric. With the two anchors above lifted, the
            dimension average moved from 0.75 to 0.80, and the original
            0.55 holistic was double-discounting the same signals the
            new anchors now carry. The 0.65 vs 0.80 gap captures the
            labeller&apos;s residual discount for the absolute size of
            the B2B revenue surface relative to the consumer business,
            which the 0.75 anchor does not fully express. Routing stays{" "}
            <code className={styles.code}>propose</code>.
          </dd>
          <dt>
            <strong>2026-05-12 ·</strong> Hook coherence judge · 1-5
            Likert → binary pass/fail with critique
          </dt>
          <dd>
            The plan originally specified a 1-5 rubric for hook
            coherence. Cross-checked against the LLM-as-judge evaluation
            literature, which is categorical that Likert scales produce
            false confidence and that the critique is where the real
            information lives. Replaced with a binary{" "}
            <code className={styles.code}>passes</code> field plus a
            mandatory{" "}
            <code className={styles.code}>critique</code> that names
            the specific evidence. Schema version bumped 2 → 3.
          </dd>
          <dt>
            <strong>2026-05-12 ·</strong> Robustness perturbations ·
            paraphrase + field_shuffle → sentence_reorder
          </dt>
          <dd>
            The plan listed{" "}
            <code className={styles.code}>typos</code>,{" "}
            <code className={styles.code}>paraphrase</code>,{" "}
            <code className={styles.code}>field_shuffle</code>, and{" "}
            <code className={styles.code}>injection</code> as the
            perturbation set. <code className={styles.code}>paraphrase</code>{" "}
            was replaced with{" "}
            <code className={styles.code}>sentence_reorder</code> (swaps
            neighbouring sentences without editing inside them) because
            paraphrase requires a per-item LLM call that would dominate
            eval cost and could itself drift; the substitution preserves
            source-quote substrings exactly.{" "}
            <code className={styles.code}>field_shuffle</code> was
            dropped: the inputs are already a profile + an optional
            company block, so &ldquo;shuffling&rdquo; them on a 2-field
            schema either no-ops or destroys the input. An LLM-driven
            paraphrase variant remains on the v1.1 list.
          </dd>
          <dt>
            <strong>2026-05-14 ·</strong> Anthropic grounding judge ·
            Claude Opus 4.7 → Claude Sonnet 4.6
          </dt>
          <dd>
            The plan specified Claude Opus 4.7 as the Anthropic-side
            grounding judge. Swapped to Claude Sonnet 4.6 during the
            cost-reduction pass: at the volume the nightly eval runs
            (73 items × claims-per-item × 2 modes), Opus pricing was
            the largest non-inference line item without a measurable
            quality lift over Sonnet on this judging task. Cohen&apos;s
            kappa with the OpenAI judge stayed in the same range
            after the swap, so the conservative-read headline (lower
            of the two grounding rates) is unaffected.
          </dd>
        </dl>
        <p>
          The cadence going forward: when a rubric edit affects a gold
          label that has already been applied, the affected items are
          re-labelled at the same time, the change is dated here, and the
          pre-fix snapshots stay in{" "}
          <code className={styles.code}>data/eval_runs/</code> so the
          delta is visible on the timeline rather than silently rebased
          into the trend.
        </p>
      </section>

      <section id="deterministic" className={styles.section}>
        <h2>6. Deterministic metrics</h2>
        <p>No LLM judge required. Computed from gold labels and outputs.</p>
        <ul>
          <li>
            <strong>Field extraction accuracy:</strong> per-field exact or
            normalised match. Industry and segment use loose token-overlap
            (50%+ of gold tokens present in the prediction); seniority and
            company_size are exact match against the controlled vocabularies.
          </li>
          <li>
            <strong>Classification accuracy:</strong> overall match across
            all four classification fields.
          </li>
          <li>
            <strong>ICP fit correlation:</strong> Pearson and Spearman of
            predicted vs gold <code className={styles.code}>fit_score.value</code>
            . Per-dimension correlation reported separately.
          </li>
          <li>
            <strong>Action accuracy:</strong> does the model pick the gold
            action given the gold thresholds?
          </li>
          <li>
            <strong>Refuse-when-should:</strong> count of items where gold is{" "}
            <code className={styles.code}>refuse</code> and the model agreed.
          </li>
          <li>
            <strong>Substring grounding:</strong> per claim, does the model&apos;s{" "}
            <code className={styles.code}>source_quote</code> appear verbatim
            in the input (whitespace-normalised, case-insensitive)? A
            substring miss is automatic ungrounded, so no judge call runs.
          </li>
          <li>
            <strong>Latency p50, p95.</strong> Per-call wall clock measured
            inside the per-mode concurrency semaphore, so the metric reflects
            model-call time and excludes time an item spent queued waiting
            for a free slot. For the multi-turn chat path, latency is the
            sum of per-turn in-semaphore wall times; harness-internal
            extractor checks between turns are not counted, since a single
            user hitting a chat endpoint wouldn&apos;t see them. The
            scorecard&apos;s headline chat latency is the corrected value
            from a 2026-05-16 re-measure; the robustness panel&apos;s chat
            latencies were not re-measured and remain on the pre-fix basis
            (the scorecard does not display robustness latency).
          </li>
          <li>
            <strong>Token cost p50, p95.</strong> Tokens in/out per call.
          </li>
          <li>
            <strong>Steps-to-completion:</strong> integrated tool-call count
            (always 1 by schema). Chat runs up to three user turns: a
            generic follow-up (&ldquo;please fill in the missing
            fields&rdquo;) is sent whenever the Haiku extractor reports
            incomplete coverage. Cap-hit at three turns is scored as
            failure. The headline exposes <em>avg_turns_used</em> and{" "}
            <em>cap_hit_rate</em> alongside the extractor-complete rate.
          </li>
        </ul>
      </section>

      <section id="judges" className={styles.section}>
        <h2>7. LLM-as-judge metrics</h2>
        <p>
          Two metrics are judged. Both are disclosed inline on the scorecard
          so prospects know which numbers are deterministic and which carry
          judge subjectivity.
        </p>
        <h3>Claim grounding</h3>
        <p>
          For every claim, the deterministic substring check runs first; a
          miss auto-fails without a judge call. Claims that pass the
          substring check go to <strong>two judges</strong>: Claude Sonnet 4.6
          and OpenAI&apos;s strongest available flagship (GPT-5 when present;
          otherwise the model name is set via{" "}
          <code className={styles.code}>OPENAI_GROUNDING_MODEL</code>). The
          scorecard publishes three values: Sonnet grounding rate, OpenAI
          judge grounding rate, and Cohen&apos;s kappa between them. The{" "}
          <em>lower of the two rates</em> is the headline number, which
          gives prospects the conservative read.
        </p>
        <h3>Hook pass rate</h3>
        <p>
          Binary pass/fail with a written critique, single judge
          (GPT-5-mini). There is no Likert scale. The literature on
          LLM-as-judge (Husain, Shankar) is consistent that 1-5 numeric
          rubrics create false confidence and the critique is where the
          learning lives.
          Pass criteria are baked into the prompt so re-runs are stable.
        </p>
        <p>
          A hook <strong>passes</strong> only when all of these hold:
          multiple specifics that come from the lead&apos;s input (verbatim
          or paraphrased; invented specifics do not count), professional
          tone appropriate for B2B outreach, on-topic and coherent.
        </p>
        <p>
          A hook <strong>fails</strong> when any of these hold: incoherent,
          off-topic, or generic with no input-grounded specifics; a single
          specific or specifics that are not actually in the input;
          over-familiar / salesy / dismissive tone; or the action is{" "}
          <code className={styles.code}>discard</code> or{" "}
          <code className={styles.code}>refuse</code> (no hook should have
          been drafted at all).
        </p>
        <p>
          The critique is required for both outcomes, since it names the
          specific evidence in the input that the hook did or did not
          ground in. A pass without a substantive critique is not a real
          pass; those land in the scorecard&apos;s judgements list for
          re-review.
        </p>
        <h3>Reading judge-mediated metrics across modes</h3>
        <p>
          GPT-5 is meaningfully stricter than the Sonnet judge on integrated
          (10-25pp lower grounding rate; less so on chat). Integrated
          produces more claims with more specificity, including
          canonical-label normalizations like
          &quot;B2B SaaS workflow automation&quot;, and GPT-5 marks more
          of them insufficiently supported by the source quote. Since the headline takes the lower of the two,
          this can pull integrated&apos;s headline grounding below
          chat&apos;s on the same run. Cohen&apos;s kappa in the 0.2-0.4
          range is itself the signal: the judges genuinely disagree on
          borderline cases.
        </p>
        <p>
          Hook pass rate is scored only over items where a hook was
          extractable. Chat refusals on adversarial inputs (jailbreak,
          fake rubric) produce no structured hook to judge and drop out
          of the chat denominator, which can inflate chat&apos;s apparent
          pass rate. The deterministic metrics (classification accuracy,
          action accuracy, substring grounding, per-field accuracy)
          score every item including refusals, so they&apos;re the
          load-bearing cross-mode comparison. The judge-mediated metrics
          report what the scorable subset looks like.
        </p>
        <p className={styles.callout}>
          Live inference is Anthropic-only. Cross-provider judging is
          offline-only, run in the eval harness for the scorecard.
        </p>
      </section>

      <section id="robustness" className={styles.section}>
        <h2>8. Robustness methodology</h2>
        <p>
          Each base item gets three perturbed variants. All perturbations are
          deterministic (seeded by item id) so the same eval run is
          reproducible from the dataset alone.
        </p>
        <ul>
          <li>
            <strong>typos:</strong> ~6% per-word character noise (swap, drop,
            duplicate).
          </li>
          <li>
            <strong>sentence_reorder:</strong> swaps neighbouring sentences
            inside paragraphs. Preserves source quote substrings by never
            editing inside a sentence. An LLM-driven paraphrase variant is
            on the v1.1 list; the current label reflects what the
            perturbation actually does.
          </li>
          <li>
            <strong>injection:</strong> appends a generic instruction-override
            probe to the company text (or profile when no company).
          </li>
        </ul>
        <p>
          Reported drops are in classification accuracy, fit-score
          correlation, and claim-grounding rate vs the main pass. The
          injection variant is the most likely to surface a real failure
          mode for the integrated build. A model citing injection text in
          a claim quote passes the substring check while still being
          semantically ungrounded. The eval-and-fix loop on the scorecard
          surfaces incidents like this with both pre-fix and post-fix
          snapshots committed.
        </p>
      </section>

      <section id="cross-mode" className={styles.section}>
        <h2>9. Cross-mode comparison framing</h2>
        <p>
          The scorecard is a <strong>bundle-vs-bundle</strong> comparison.
          The two builds differ in four ways at once:
        </p>
        <ul>
          <li>
            <strong>Output contract:</strong> strict{" "}
            <code className={styles.code}>enrich_lead</code> tool schema
            (integrated) vs free-form prose (chat).
          </li>
          <li>
            <strong>Grounding requirement:</strong> per-claim{" "}
            <code className={styles.code}>source_quote</code> required by the
            schema (integrated) vs no structural requirement (chat).
          </li>
          <li>
            <strong>Inference shape:</strong> single structured call
            (integrated) vs multi-turn chat plus a Haiku 4.5 extractor pass
            that maps prose back to the schema (chat).
          </li>
          <li>
            <strong>Extended thinking:</strong> 4000-token budget on
            (integrated) vs off (chat).
          </li>
        </ul>
        <p>
          The scorecard treats these as one bundle because that is what a
          product team actually ships. Chat surfaces do not typically expose
          thinking deltas in the UI, since the affordance does not fit.
          Structured backends often do. The realistic comparison is
          product-surface-A vs
          product-surface-B, not contract-vs-contract under matched compute.
          The chat build is polished to the same product standard as
          integrated; neither side is a strawman.
        </p>
        <p>
          A stricter version of this question (&ldquo;does the schema
          discipline win even with thinking equal?&rdquo;) would run both
          modes with thinking matched (both on, or both off). That is a
          separate, sharper claim that the current eval does not make. It is
          on the roadmap as a second panel; the result would either reinforce
          the contract argument or expose how much of the delta is thinking.
        </p>
        <p>
          The chat build&apos;s free-form output is parsed back to the
          structured schema by the Haiku 4.5 extractor pass mentioned above,
          so both modes score against the same gold labels. The extractor
          sees the chat output only, never the original input, so it
          cannot hallucinate grounding. If the chat reply omitted a source quote,
          no extracted claim has one either.
        </p>
      </section>

      <section id="ops" className={styles.section}>
        <h2>10. Operations</h2>
        <ul>
          <li>
            Eval runner: Python script in{" "}
            <code className={styles.code}>services/eval/</code>. Live
            (non-batch) inference with bounded asyncio concurrency for true
            per-call latency. Batch API is available as a future toggle when
            test-set size grows.
          </li>
          <li>
            Regression gate runs on every PR that touches eval-relevant
            paths (cheap anchor subset, no judges). Canonical scorecard
            snapshots refresh on demand via{" "}
            <code className={styles.code}>workflow_dispatch</code>, not on
            a fixed schedule. The demo doesn&apos;t ship daily, so a
            time-based cron mostly re-runs identical experiments.
          </li>
          <li>
            Results commit as a JSON snapshot to{" "}
            <code className={styles.code}>data/eval_runs/latest.json</code>{" "}
            plus a timestamped historical copy.
          </li>
          <li>
            Inter-judge Cohen&apos;s kappa for grounding is reported each
            run.
          </li>
          <li>
            Methodology page names every metric, sample size, and judge model
            used.
          </li>
        </ul>
      </section>

      <section id="telemetry" className={styles.section}>
        <h2>11. Telemetry vs adoption</h2>
        <p>
          The demo instruments prospect-funnel signals (own-input-pasted,
          scorecard-clicked-from-result, page-bounce-after-low-fit,
          contact-link-clicked-from-result). Funnel data is not adoption
          data. Real adoption measurement requires real users on a deployed
          feature over time. The eval scorecard is a cold benchmark on a
          static test set. It tells you how the system performs on
          labelled examples, not how it performs on your actual pipeline.
        </p>
      </section>

      <section id="production" className={styles.section}>
        <h2>12. Input shape in production</h2>
        <p>
          The demo accepts pasted text. Real deployments are triggered by a
          CRM record landing, a browser extension on LinkedIn, or a CSV bulk
          import. The paste interface is a deployable proxy that exercises
          the same model contract; wiring the trigger to a CRM is the
          v2/services hook.
        </p>
      </section>

      <section id="limits" className={styles.section}>
        <h2>13. Known limits</h2>
        <ul>
          <li>
            <strong>Synthetic test set.</strong> 73 hand-labelled items;
            generalisation to real pipelines is unmeasured.
          </li>
          <li>
            <strong>Single-labeller bias.</strong> Solo-labelled in v1.
            v1.1 adds a peer second-pass on fit-score and claims-allowed
            labels.
          </li>
          <li>
            <strong>Judge subjectivity</strong> on the hook pass/fail
            decision. Single judge by design; binary outcome plus
            mandatory critique limits drift but doesn&apos;t eliminate
            it. Hook judgements with thin critiques are surfaced in the
            judgements list for re-review.
          </li>
          <li>
            <strong>Fictional ICP.</strong> Real customer ICPs differ; the
            structured schema accepts custom ICPs (see below) but the
            current scorecard is anchored to this one.
          </li>
          <li>
            <strong>Supported-language scope.</strong> English, Swedish,
            German. Other languages return an out-of-scope response.
          </li>
          <li>
            <strong>No adoption data.</strong> Funnel telemetry exists;
            usage measurement on a deployed product does not.
          </li>
        </ul>
      </section>

      <section id="custom-icp" className={styles.section}>
        <h2>14. Custom ICP, v2 hook</h2>
        <p>
          v1 ships a fixed ICP. A v2 deployment accepts a customer-supplied
          ICP. Two shapes the schema already supports:
        </p>
        <ol>
          <li>
            <strong>Structured ICP</strong> with concrete ranges for stage,
            headcount, and ARR plus natural-language predicates for product
            shape and target roles. Structured fields run through
            deterministic range checks; predicates run through anchored
            0/0.25/0.5/0.75/1.0 rubrics that the model evaluates per call.
            Customers calibrate rubric anchors against five to ten labelled
            examples from their own pipeline.
          </li>
          <li>
            <strong>Hybrid ICP</strong> that pulls structured fields from the
            customer&apos;s CRM (HubSpot, Salesforce, Attio) and uses
            predicates for soft criteria. Rubric anchors get versioned
            alongside the predicate text so changes are auditable.
          </li>
        </ol>
        <p className={styles.callout}>
          v2 is where this becomes a feature wired into a customer&apos;s
          revenue stack. Book a call from the homepage if you want that
          conversation.
        </p>
      </section>
    </main>
  );
}
