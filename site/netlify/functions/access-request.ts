import type { Context } from "@netlify/functions";
import { verifyGoogleIdToken } from "./_shared/google.js";
import { isAllowed, addRequest, appendAudit, nowIso, PENDING_LIMIT } from "./_shared/blobs.js";
import { checkOrigin } from "./_shared/origin-check.js";

export default async (req: Request, _ctx: Context): Promise<Response> => {
  if (req.method !== "POST") return new Response("Method Not Allowed", { status: 405 });
  const originErr = checkOrigin(req);
  if (originErr) return originErr;

  let body: { idToken?: string; message?: string };
  try {
    body = await req.json();
  } catch {
    return json({ error: "bad_json" }, 400);
  }
  if (!body.idToken) return json({ error: "missing_id_token" }, 400);

  const claims = await verifyGoogleIdToken(body.idToken);
  if (!claims) return json({ error: "invalid_token" }, 401);

  if (await isAllowed(claims.email)) {
    return json({ status: "already_allowed" });
  }

  const message = typeof body.message === "string" ? body.message.slice(0, 500) : undefined;
  const result = await addRequest(claims.email, {
    requestedAt: nowIso(),
    message,
    name: claims.name,
  });

  if (!result.ok && result.reason === "limit") {
    return json({ error: "pending_limit", limit: PENDING_LIMIT }, 429);
  }

  await appendAudit({
    at: nowIso(),
    actor: claims.email,
    action: "request_access",
    detail: message ? { hasMessage: true } : undefined,
  });

  return json({ status: "submitted" });
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
