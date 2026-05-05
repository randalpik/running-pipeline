import type { Context } from "@netlify/functions";
import { requireAdmin } from "./_shared/admin-guard.js";
import { getRequests } from "./_shared/blobs.js";

export default async (req: Request, _ctx: Context): Promise<Response> => {
  const guard = await requireAdmin(req);
  if (guard instanceof Response) return guard;

  const requests = await getRequests();
  return new Response(JSON.stringify({ requests }), {
    status: 200,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
};
