import type { Context } from "@netlify/functions";
import { verifyGoogleIdToken } from "./_shared/google.js";
import { isAllowed, addRequest, appendAudit, nowIso, PENDING_LIMIT } from "./_shared/blobs.js";
import { checkOrigin } from "./_shared/origin-check.js";
import { sendEmail, escapeHtml } from "./_shared/mailer.js";

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

  const mailStatus = await notifyAdmin(claims.email, claims.name, message);

  await appendAudit({
    at: nowIso(),
    actor: claims.email,
    action: "request_access",
    detail: { hasMessage: !!message, mail: mailStatus },
  });

  return json({ status: "submitted" });
};

async function notifyAdmin(
  requesterEmail: string,
  requesterName: string | undefined,
  message: string | undefined
): Promise<string> {
  const adminEmail = process.env.ADMIN_EMAIL;
  if (!adminEmail) return "skipped:no_admin";

  const displayName = requesterName?.trim() || requesterEmail;
  const messageBlockHtml = message
    ? `<p style="margin:16px 0 8px">Message:</p>
       <blockquote style="margin:0 0 16px;border-left:3px solid #ccc;padding:8px 12px;color:#444;background:#f7f7f7;">${escapeHtml(message)}</blockquote>`
    : `<p style="color:#777">No message included.</p>`;
  const messageBlockText = message ? `Message:\n${message}\n` : `No message included.\n`;

  const html = `
<p><strong>${escapeHtml(displayName)}</strong> &lt;${escapeHtml(requesterEmail)}&gt; requested access.</p>
${messageBlockHtml}
<p><a href="https://running.maxrandalmusic.com/admin.html">Review on the admin page →</a></p>
`.trim();

  const text = `${displayName} <${requesterEmail}> requested access.

${messageBlockText}
Review: https://running.maxrandalmusic.com/admin.html
`;

  const result = await sendEmail({
    to: adminEmail,
    subject: `Access request from ${displayName} <${requesterEmail}>`,
    html,
    text,
  });
  return result.ok ? "sent" : `failed:${result.error ?? "unknown"}`;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
