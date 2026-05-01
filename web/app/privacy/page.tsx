export default function PrivacyPage() {
  return (
    <main>
      <h1>Privacy</h1>
      <p>
        Free-form inputs are not stored by us. Inputs are sent to Anthropic for
        inference under their API data policy (Anthropic does not train on API
        inputs by default). Plausible is used for cookieless analytics. Sentry
        may log IP addresses on error. Upstash logs IPs for rate limiting.
      </p>
    </main>
  );
}
