import { NextResponse } from "next/server";
import { modalUrl } from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = await req.text();
  const upstream = await fetch(modalUrl("/neighbours"), {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-forwarded-for":
        req.headers.get("x-forwarded-for") ??
        req.headers.get("x-real-ip") ??
        "",
    },
    body,
  });
  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "content-type": "application/json" },
  });
}
