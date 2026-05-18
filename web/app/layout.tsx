import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";
import { SiteNav } from "./site-nav";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-body",
});

export const metadata: Metadata = {
  title: "Lead Enrichment",
  description:
    "A B2B lead enrichment AI feature, built two ways. Same model, different output contracts. Live demo and eval scorecard.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <div className="best-on-desktop">
          Best on desktop, but the demo still works on mobile.
        </div>
        <SiteNav />
        {children}
        <Analytics />
      </body>
    </html>
  );
}
