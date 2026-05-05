import type { Context } from "@netlify/functions";
import { verifyGoogleIdToken } from "./_shared/google.js";
import { isAllowed } from "./_shared/blobs.js";
import { mintSession, sessionCookie } from "./_shared/session.js";
import { isAdminEmail } from "./_shared/admin-guard.js";
import { checkOrigin } from "./_shared/origin-check.js";

export default async (req: Request, _ctx: Context): Promise<Response> => {
  if (req.method !== "POST") return new Response("Method Not Allowed", { status: 405 });
  const originErr = checkOrigin(req);
  if (originErr) return originErr;

  let body: { idToken?: string };
  try {
    body = await req.json();
  } catch {
    return json({ error: "bad_json" }, 400);
  }
  if (!body.idToken) return json({ error: "missing_id_token" }, 400);

  const claims = await verifyGoogleIdToken(body.idToken);
  if (!claims) return json({ error: "invalid_token" }, 401);

  const allowed = await isAllowed(claims.email);
  if (!allowed) {
    return json({ status: "unallowed", email: claims.email, name: claims.name ?? null });
  }

  const token = await mintSession(claims.email);
  return json(
    { status: "allowed", email: claims.email, isAdmin: isAdminEmail(claims.email) },
    200,
    { "Set-Cookie": sessionCookie(token) }
  );
};

function json(body: unknown, status = 200, extraHeaders: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
      ...extraHeaders,
    },
  });
}
