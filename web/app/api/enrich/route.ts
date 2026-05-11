import { modalUrl } from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = await req.text();
  const upstream = await fetch(modalUrl("/enrich"), {
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
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}
