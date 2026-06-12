import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export function GET() {
  const apiUrl = process.env.NEXT_PUBLIC_MDNAC_API_URL || null;
  const wsUrl = process.env.MDNAC_PUBLIC_WS_URL || process.env.NEXT_PUBLIC_MDNAC_WS_URL || null;

  return NextResponse.json({
    apiUrl,
    wsUrl,
  });
}
