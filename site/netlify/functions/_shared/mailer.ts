// Resend transactional email helper.
//
// Reads RESEND_API_KEY and MAIL_FROM from process.env. If either is unset,
// sendEmail is a no-op that returns { ok: false, error: 'not_configured' }
// — so deploys done before Resend is wired up don't fail; mail just
// silently doesn't go and the audit log records 'failed:not_configured'.
//
// Errors never throw. Callers should treat mail as best-effort and not
// gate any user-facing or admin action on the result.

export interface MailOpts {
  to: string;
  subject: string;
  html: string;
  text?: string;
  replyTo?: string;
}

export interface MailResult {
  ok: boolean;
  error?: string;
}

export async function sendEmail(opts: MailOpts): Promise<MailResult> {
  const apiKey = process.env.RESEND_API_KEY;
  const from = process.env.MAIL_FROM;
  if (!apiKey || !from) {
    console.warn("Mailer: RESEND_API_KEY or MAIL_FROM not set; skipping send to", opts.to);
    return { ok: false, error: "not_configured" };
  }

  const body: Record<string, unknown> = {
    from,
    to: [opts.to],
    subject: opts.subject,
    html: opts.html,
  };
  if (opts.text) body.text = opts.text;
  if (opts.replyTo) body.reply_to = opts.replyTo;

  try {
    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      return { ok: false, error: `resend_${res.status}: ${detail.slice(0, 200)}` };
    }
    return { ok: true };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, error: msg };
  }
}

export function escapeHtml(s: string): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
