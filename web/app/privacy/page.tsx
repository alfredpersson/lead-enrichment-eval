export default function PrivacyPage() {
  return (
    <main>
      <h1>Privacy</h1>
      <ul>
        <li>
          Free-form profile and company inputs are not stored on our servers.
        </li>
        <li>
          Inputs are sent to Anthropic for inference under their API data
          policy (Anthropic does not train on API inputs by default).
        </li>
        <li>
          Chat conversations are stored in your browser&rsquo;s local storage
          and never leave your device, so you can return to past leads.
          Clear them with the delete control on each conversation in the
          chat sidebar, or by clearing your browser data.
        </li>
        <li>Plausible is used for cookieless analytics.</li>
        <li>Sentry may log IP addresses on error.</li>
        <li>Upstash logs IPs for rate limiting.</li>
      </ul>
    </main>
  );
}
