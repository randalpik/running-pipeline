import type { Context } from "@netlify/functions";
import { requireAdmin } from "./_shared/admin-guard.js";
import { addToAllowlist, removeRequest, appendAudit, nowIso } from "./_shared/blobs.js";
import { checkOrigin } from "./_shared/origin-check.js";

export default async (req: Request, _ctx: Context): Promise<Response> => {
  if (req.method !== "POST") return new Response("Method Not Allowed", { status: 405 });
  const originErr = checkOrigin(req);
  if (originErr) return originErr;
  const guard = await requireAdmin(req);
  if (guard instanceof Response) return guard;

  let body: { email?: string };
  try {
    body = await req.json();
  } catch {
    return json({ error: "bad_json" }, 400);
  }
  const email = body.email?.toLowerCase().trim();
  if (!email || !email.includes("@")) return json({ error: "bad_email" }, 400);

  await addToAllowlist(email);
  await removeRequest(email);
  await appendAudit({ at: nowIso(), actor: guard.email, action: "approve", target: email });

  return json({ status: "approved", email });
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
