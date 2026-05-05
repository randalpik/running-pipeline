import type { Context } from "@netlify/functions";
import { getSession } from "./_shared/session.js";
import { isAdminEmail } from "./_shared/admin-guard.js";
import { isAllowed } from "./_shared/blobs.js";

export default async (req: Request, _ctx: Context): Promise<Response> => {
  const session = await getSession(req);
  if (!session) return json({ status: "anonymous" }, 401);

  const allowed = await isAllowed(session.email);
  if (!allowed) {
    return json({ status: "revoked", email: session.email }, 403);
  }
  return json({
    status: "allowed",
    email: session.email,
    isAdmin: isAdminEmail(session.email),
  });
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
  });
}
