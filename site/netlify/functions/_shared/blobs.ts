import { getStore, type Store } from "@netlify/blobs";

export interface AccessRequest {
  requestedAt: string;
  message?: string;
  name?: string;
}

export interface AuditEntry {
  at: string;
  actor: string;
  action: string;
  target?: string;
  detail?: Record<string, unknown>;
}

const KEY = "data";
const MAX_AUDIT_ENTRIES = 1000;
const MAX_PENDING_REQUESTS = 50;

export const PENDING_LIMIT = MAX_PENDING_REQUESTS;

function store(name: "allowlist" | "requests" | "audit"): Store {
  return getStore({ name, consistency: "strong" });
}

async function readJson<T>(s: Store, fallback: T): Promise<T> {
  const data = (await s.get(KEY, { type: "json" })) as T | null;
  return data ?? fallback;
}

async function mutateJson<T>(
  s: Store,
  fallback: T,
  fn: (v: T) => T | Promise<T>
): Promise<T> {
  const current = await readJson<T>(s, fallback);
  const next = await fn(current);
  await s.setJSON(KEY, next);
  return next;
}

export async function getAllowlist(): Promise<string[]> {
  return readJson<string[]>(store("allowlist"), []);
}

export async function isAllowed(email: string): Promise<boolean> {
  const e = email.toLowerCase();
  const admin = process.env.ADMIN_EMAIL?.toLowerCase();
  if (admin && e === admin) return true;
  const list = await getAllowlist();
  return list.includes(e);
}

export async function addToAllowlist(email: string): Promise<string[]> {
  const e = email.toLowerCase();
  return mutateJson<string[]>(store("allowlist"), [], (list) => {
    if (list.includes(e)) return list;
    return [...list, e].sort();
  });
}

export async function removeFromAllowlist(email: string): Promise<string[]> {
  const e = email.toLowerCase();
  return mutateJson<string[]>(store("allowlist"), [], (list) =>
    list.filter((x) => x !== e)
  );
}

export async function getRequests(): Promise<Record<string, AccessRequest>> {
  return readJson<Record<string, AccessRequest>>(store("requests"), {});
}

export async function addRequest(
  email: string,
  req: AccessRequest
): Promise<{ ok: boolean; reason?: string }> {
  const e = email.toLowerCase();
  let reason: string | undefined;
  await mutateJson<Record<string, AccessRequest>>(store("requests"), {}, (current) => {
    if (Object.keys(current).length >= MAX_PENDING_REQUESTS && !(e in current)) {
      reason = "limit";
      return current;
    }
    return { ...current, [e]: req };
  });
  return reason ? { ok: false, reason } : { ok: true };
}

export async function removeRequest(email: string): Promise<Record<string, AccessRequest>> {
  const e = email.toLowerCase();
  return mutateJson<Record<string, AccessRequest>>(store("requests"), {}, (current) => {
    if (!(e in current)) return current;
    const { [e]: _, ...rest } = current;
    return rest;
  });
}

export async function appendAudit(entry: AuditEntry): Promise<void> {
  await mutateJson<AuditEntry[]>(store("audit"), [], (list) => {
    const next = [...list, entry];
    return next.length > MAX_AUDIT_ENTRIES ? next.slice(-MAX_AUDIT_ENTRIES) : next;
  });
}

export function nowIso(): string {
  return new Date().toISOString();
}
