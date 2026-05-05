import type { Context } from "@netlify/functions";
import { requireAdmin } from "./_shared/admin-guard.js";
import {
  addToAllowlist,
  removeRequest,
  appendAudit,
  getRequests,
  nowIso,
} from "./_shared/blobs.js";
import { checkOrigin } from "./_shared/origin-check.js";
import { sendEmail, escapeHtml } from "./_shared/mailer.js";

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

  // Read the pending request before clearing it, so we have the requester's
  // name available for the approval email's greeting.
  const requests = await getRequests();
  const request = requests[email] || null;

  await addToAllowlist(email);
  await removeRequest(email);

  const mailStatus = await notifyRequester(email, request?.name);

  await appendAudit({
    at: nowIso(),
    actor: guard.email,
    action: "approve",
    target: email,
    detail: { mail: mailStatus },
  });

  return json({ status: "approved", email });
};

async function notifyRequester(
  email: string,
  name: string | undefined
): Promise<string> {
  const replyTo = process.env.ADMIN_REPLY_TO;
  const firstName = (name?.split(" ")[0] || "").trim();
  const greeting = firstName ? `Hi ${firstName},` : "Hi,";

  const html = `
<p>${escapeHtml(greeting)}</p>
<p>Max approved your access to <a href="https://running.maxrandalmusic.com">running.maxrandalmusic.com</a>.</p>
<p>You can sign in now using the same Google account you used to request access.</p>
`.trim();

  const text = `${greeting}

Max approved your access to running.maxrandalmusic.com.

You can sign in now using the same Google account you used to request access.
`;

  const result = await sendEmail({
    to: email,
    subject: "You're in — Max's Running Data",
    html,
    text,
    replyTo: replyTo || undefined,
  });
  return result.ok ? "sent" : `failed:${result.error ?? "unknown"}`;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
