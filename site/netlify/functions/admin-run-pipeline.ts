import type { Context } from "@netlify/functions";
import { requireAdmin } from "./_shared/admin-guard.js";
import { appendAudit, nowIso } from "./_shared/blobs.js";
import { checkOrigin } from "./_shared/origin-check.js";

const WORKFLOW_FILE = "build-and-deploy.yml";

export default async (req: Request, _ctx: Context): Promise<Response> => {
  if (req.method !== "POST") return new Response("Method Not Allowed", { status: 405 });
  const originErr = checkOrigin(req);
  if (originErr) return originErr;
  const guard = await requireAdmin(req);
  if (guard instanceof Response) return guard;

  const repo = process.env.GITHUB_REPO;
  const token = process.env.GITHUB_DISPATCH_TOKEN;
  if (!repo || !token) return json({ error: "github_not_configured" }, 500);

  let body: { fit?: boolean; ref?: string };
  try {
    body = await req.json();
  } catch {
    body = {};
  }
  const ref = body.ref || "main";
  const inputs = { fit: body.fit ? "true" : "false" };

  const dispatchRes = await fetch(
    `https://api.github.com/repos/${repo}/actions/workflows/${WORKFLOW_FILE}/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref, inputs }),
    }
  );

  if (!dispatchRes.ok) {
    const detail = await dispatchRes.text().catch(() => "");
    return json(
      { error: "dispatch_failed", status: dispatchRes.status, detail: detail.slice(0, 500) },
      502
    );
  }

  await appendAudit({
    at: nowIso(),
    actor: guard.email,
    action: "run_pipeline",
    detail: { fit: !!body.fit, ref },
  });

  const runUrl = await pollForRunUrl(repo, token);
  return json({ status: "dispatched", runUrl });
};

async function pollForRunUrl(repo: string, token: string): Promise<string | null> {
  const since = new Date(Date.now() - 60_000).toISOString();
  for (let i = 0; i < 5; i++) {
    await new Promise((r) => setTimeout(r, 1000 * (i + 1)));
    const res = await fetch(
      `https://api.github.com/repos/${repo}/actions/runs?event=workflow_dispatch&per_page=5&created=>${encodeURIComponent(since)}`,
      {
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
        },
      }
    );
    if (!res.ok) continue;
    const data = (await res.json()) as { workflow_runs?: Array<{ html_url: string }> };
    if (data.workflow_runs && data.workflow_runs.length > 0) {
      return data.workflow_runs[0].html_url;
    }
  }
  return null;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
