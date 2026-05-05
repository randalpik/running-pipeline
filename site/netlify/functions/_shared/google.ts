import { jwtVerify, createRemoteJWKSet } from "jose";

const JWKS = createRemoteJWKSet(new URL("https://www.googleapis.com/oauth2/v3/certs"));

export interface GoogleIdClaims {
  email: string;
  email_verified: boolean;
  sub: string;
  name?: string;
  picture?: string;
}

export async function verifyGoogleIdToken(idToken: string): Promise<GoogleIdClaims | null> {
  const clientId = process.env.GOOGLE_OAUTH_CLIENT_ID;
  if (!clientId) throw new Error("GOOGLE_OAUTH_CLIENT_ID not set");
  try {
    const { payload } = await jwtVerify(idToken, JWKS, {
      audience: clientId,
      issuer: ["https://accounts.google.com", "accounts.google.com"],
    });
    if (
      typeof payload.email !== "string" ||
      payload.email_verified !== true ||
      typeof payload.sub !== "string"
    ) {
      return null;
    }
    return {
      email: payload.email.toLowerCase(),
      email_verified: true,
      sub: payload.sub,
      name: typeof payload.name === "string" ? payload.name : undefined,
      picture: typeof payload.picture === "string" ? payload.picture : undefined,
    };
  } catch {
    return null;
  }
}
