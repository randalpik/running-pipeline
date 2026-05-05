// Auth gate for the entire site.
//
// Model: gate-everything-by-default. The function fires on /** (all paths).
// Anything not on the EXEMPT list below requires a valid session cookie
// whose email is on the allowlist (or matches ADMIN_EMAIL).
//
// To make a new path PUBLIC, add it to EXEMPT_PATHS or EXEMPT_PREFIXES below.
// Forgetting to do so is the safe failure mode (the path stays gated).
// Forgetting to gate something would be the unsafe failure mode and is
// what this allow-list-of-public-paths design avoids.

import { jwtVerify } from "https://esm.sh/jose@5.9.6";
import { getStore } from "@netlify/blobs";
import type { Context } from "@netlify/edge-functions";

const SESSION_COOKIE = "__Host-session";
const ALLOWLIST_TTL_MS = 30_000;

// Paths that bypass the gate entirely. These are publicly served.
const EXEMPT_PATHS = new Set<string>([
  "/login.html",
  "/plotly.min.js",
  "/favicon.ico",
  "/robots.txt",
]);
// Path prefixes that bypass the gate (Netlify internals + our API surface).
// /api/** is exempt because each /api/* function does its own auth check
// (auth-config and auth-exchange are public; everything else verifies a
// session or admin status server-side).
const EXEMPT_PREFIXES = ["/api/", "/.netlify/"];

let allowlistCache: { value: Set<string>; expires: number } | null = null;

async function getAllowlist(): Promise<Set<string>> {
  const now = Date.now();
  if (allowlistCache && allowlistCache.expires > now) return allowlistCache.value;
  const store = getStore({ name: "allowlist", consistency: "strong" });
  const data = (await store.get("data", { type: "json" })) as string[] | null;
  const set = new Set((data ?? []).map((e) => e.toLowerCase()));
  allowlistCache = { value: set, expires: now + ALLOWLIST_TTL_MS };
  return set;
}

function readSessionCookie(cookieHeader: string | null): string | null {
  if (!cookieHeader) return null;
  for (const part of cookieHeader.split(";")) {
    const [k, ...rest] = part.trim().split("=");
    if (k === SESSION_COOKIE) return rest.join("=");
  }
  return null;
}

async function verifySession(token: string, secretStr: string): Promise<{ email: string } | null> {
  try {
    const secret = new TextEncoder().encode(secretStr);
    const { payload } = await jwtVerify(token, secret);
    if (typeof payload.email !== "string") return null;
    return { email: payload.email.toLowerCase() };
  } catch {
    return null;
  }
}

function isExempt(pathname: string): boolean {
  if (EXEMPT_PATHS.has(pathname)) return true;
  for (const p of EXEMPT_PREFIXES) {
    if (pathname.startsWith(p)) return true;
  }
  return false;
}

export default async (req: Request, _ctx: Context): Promise<Response | void> => {
  const url = new URL(req.url);
  if (isExempt(url.pathname)) return;

  const secretStr = Deno.env.get("SESSION_JWT_SECRET");
  if (!secretStr) return new Response("Misconfigured", { status: 500 });

  const token = readSessionCookie(req.headers.get("cookie"));
  const session = token ? await verifySession(token, secretStr) : null;

  let allowed = false;
  if (session) {
    const admin = Deno.env.get("ADMIN_EMAIL")?.toLowerCase();
    if (admin && session.email === admin) {
      allowed = true;
    } else {
      const list = await getAllowlist();
      allowed = list.has(session.email);
    }
  }

  if (allowed) return;

  // For root, omit `next` — the default post-login destination is /.
  if (url.pathname === "/") {
    return Response.redirect(new URL("/login.html", url.origin).toString(), 302);
  }
  const next = url.pathname + url.search;
  const loginUrl = `/login.html?next=${encodeURIComponent(next)}`;
  return Response.redirect(new URL(loginUrl, url.origin).toString(), 302);
};

export const config = { path: ["/", "/*", "/**"] };
