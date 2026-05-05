import type { Context } from "@netlify/functions";
import { requireAdmin } from "./_shared/admin-guard.js";
import {
  getAllowlist,
  addToAllowlist,
  removeFromAllowlist,
  appendAudit,
  nowIso,
} from "./_shared/blobs.js";
import { checkOrigin } from "./_shared/origin-check.js";

export default async (req: Request, _ctx: Context): Promise<Response> => {
  const guard = await requireAdmin(req);
  if (guard instanceof Response) return guard;

  if (req.method === "GET") {
    const list = await getAllowlist();
    return json({ allowlist: list });
  }

  const originErr = checkOrigin(req);
  if (originErr) return originErr;

  if (req.method === "POST" || req.method === "DELETE") {
    let body: { email?: string };
    try {
      body = await req.json();
    } catch {
      return json({ error: "bad_json" }, 400);
    }
    const email = body.email?.toLowerCase().trim();
    if (!email || !email.includes("@")) return json({ error: "bad_email" }, 400);

    if (req.method === "POST") {
      const list = await addToAllowlist(email);
      await appendAudit({ at: nowIso(), actor: guard.email, action: "allowlist_add", target: email });
      return json({ allowlist: list });
    }
    const list = await removeFromAllowlist(email);
    await appendAudit({ at: nowIso(), actor: guard.email, action: "allowlist_remove", target: email });
    return json({ allowlist: list });
  }

  return new Response("Method Not Allowed", { status: 405 });
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
