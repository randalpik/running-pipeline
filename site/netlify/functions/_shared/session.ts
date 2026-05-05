import { SignJWT, jwtVerify } from "jose";

const SESSION_COOKIE = "__Host-session";
const SESSION_TTL_DAYS = 30;

function secret(): Uint8Array {
  const s = process.env.SESSION_JWT_SECRET;
  if (!s) throw new Error("SESSION_JWT_SECRET not set");
  return new TextEncoder().encode(s);
}

export interface SessionClaims {
  email: string;
  iat: number;
  exp: number;
}

export async function mintSession(email: string): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  return new SignJWT({ email })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt(now)
    .setExpirationTime(now + SESSION_TTL_DAYS * 24 * 60 * 60)
    .sign(secret());
}

export async function verifySession(token: string): Promise<SessionClaims | null> {
  try {
    const { payload } = await jwtVerify(token, secret());
    if (typeof payload.email !== "string") return null;
    return payload as unknown as SessionClaims;
  } catch {
    return null;
  }
}

export function sessionCookie(token: string): string {
  const maxAge = SESSION_TTL_DAYS * 24 * 60 * 60;
  return `${SESSION_COOKIE}=${token}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${maxAge}`;
}

export function clearSessionCookie(): string {
  return `${SESSION_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0`;
}

export function readSessionCookie(cookieHeader: string | null | undefined): string | null {
  if (!cookieHeader) return null;
  for (const part of cookieHeader.split(";")) {
    const [k, ...rest] = part.trim().split("=");
    if (k === SESSION_COOKIE) return rest.join("=");
  }
  return null;
}

export async function getSession(req: Request): Promise<SessionClaims | null> {
  const token = readSessionCookie(req.headers.get("cookie"));
  if (!token) return null;
  return verifySession(token);
}
