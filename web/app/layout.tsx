import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lead Enrichment — integrated vs. chat",
  description:
    "A B2B lead enrichment AI feature, built two ways. Same model, different output contracts. Live demo and eval scorecard.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
