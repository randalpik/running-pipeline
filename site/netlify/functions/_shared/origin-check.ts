const PROD_ORIGIN = "https://running.maxrandalmusic.com";
const DEV_ORIGIN = "http://localhost:8888";

function allowedOrigins(): string[] {
  const extra =
    process.env.ALLOWED_ORIGINS?.split(",").map((s: string) => s.trim()).filter(Boolean) ?? [];
  return [PROD_ORIGIN, DEV_ORIGIN, ...extra];
}

export function checkOrigin(req: Request): Response | null {
  const origin = req.headers.get("origin") ?? deriveOriginFromReferer(req.headers.get("referer"));
  if (!origin) return new Response("Missing Origin", { status: 403 });
  if (!allowedOrigins().includes(origin)) {
    return new Response("Bad Origin", { status: 403 });
  }
  return null;
}

function deriveOriginFromReferer(referer: string | null): string | null {
  if (!referer) return null;
  try {
    const u = new URL(referer);
    return `${u.protocol}//${u.host}`;
  } catch {
    return null;
  }
}
