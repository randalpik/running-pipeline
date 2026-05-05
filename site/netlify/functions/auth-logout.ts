import type { Context } from "@netlify/functions";
import { clearSessionCookie } from "./_shared/session.js";
import { checkOrigin } from "./_shared/origin-check.js";

export default async (req: Request, _ctx: Context): Promise<Response> => {
  if (req.method !== "POST") return new Response("Method Not Allowed", { status: 405 });
  const originErr = checkOrigin(req);
  if (originErr) return originErr;

  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
      "Set-Cookie": clearSessionCookie(),
    },
  });
};
