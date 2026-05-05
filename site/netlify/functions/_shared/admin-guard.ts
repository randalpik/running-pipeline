import { getSession, type SessionClaims } from "./session.js";

export function adminEmail(): string {
  const e = process.env.ADMIN_EMAIL;
  if (!e) throw new Error("ADMIN_EMAIL not set");
  return e.toLowerCase();
}

export function isAdminEmail(email: string): boolean {
  return email.toLowerCase() === adminEmail();
}

export async function requireAdmin(req: Request): Promise<SessionClaims | Response> {
  const session = await getSession(req);
  if (!session) return new Response("Unauthorized", { status: 401 });
  if (!isAdminEmail(session.email)) return new Response("Forbidden", { status: 403 });
  return session;
}
