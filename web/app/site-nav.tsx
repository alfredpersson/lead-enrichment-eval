"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { track } from "@/lib/analytics";

export function SiteNav() {
  const pathname = usePathname() ?? "/";
  const mode = pathname.startsWith("/chat")
    ? "chat"
    : pathname.startsWith("/integrated")
      ? "integrated"
      : null;
  return (
    <nav className="site-nav" aria-label="primary">
      <Link href="/" className="site-nav__brand">
        lead enrichment
      </Link>
      <div
        className="site-nav__toggle"
        role="group"
        aria-label="Switch between integrated and chat builds"
      >
        <Link
          href="/integrated"
          className={`site-nav__toggle-option ${
            mode === "integrated" ? "site-nav__toggle-option--active" : ""
          }`}
          aria-current={mode === "integrated" ? "page" : undefined}
          onClick={() => {
            if (mode !== "integrated") track({ name: "mode-switched", props: { to: "integrated" } });
          }}
        >
          Integrated
        </Link>
        <Link
          href="/chat"
          className={`site-nav__toggle-option ${
            mode === "chat" ? "site-nav__toggle-option--active" : ""
          }`}
          aria-current={mode === "chat" ? "page" : undefined}
          onClick={() => {
            if (mode !== "chat") track({ name: "mode-switched", props: { to: "chat" } });
          }}
        >
          Chat
        </Link>
      </div>
      <div className="site-nav__secondary" aria-label="secondary">
        <Link
          href="/scorecard"
          onClick={() => {
            if (mode) track({ name: "scorecard-clicked-from-result", props: { surface: mode } });
          }}
        >
          Scorecard
        </Link>
        <Link href="/methodology">Methodology</Link>
      </div>
    </nav>
  );
}
