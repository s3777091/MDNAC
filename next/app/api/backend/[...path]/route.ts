import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{
    path?: string[];
  }>;
};

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "content-length",
  "host",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

export async function GET(request: NextRequest, context: RouteContext) {
  return proxyBackend(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxyBackend(request, context);
}

export async function PUT(request: NextRequest, context: RouteContext) {
  return proxyBackend(request, context);
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  return proxyBackend(request, context);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return proxyBackend(request, context);
}

async function proxyBackend(request: NextRequest, context: RouteContext) {
  const { path = [] } = await context.params;
  const target = new URL(path.join("/"), normalizedBaseUrl());
  target.search = request.nextUrl.search;

  const headers = new Headers(request.headers);
  for (const header of HOP_BY_HOP_HEADERS) {
    headers.delete(header);
  }

  let body: BodyInit | undefined;
  if (!["GET", "HEAD"].includes(request.method)) {
    body = await request.arrayBuffer();
  }

  try {
    const response = await fetch(target, {
      method: request.method,
      headers,
      body,
      cache: "no-store",
    });

    const responseHeaders = new Headers(response.headers);
    for (const header of HOP_BY_HOP_HEADERS) {
      responseHeaders.delete(header);
    }

    return new NextResponse(await response.arrayBuffer(), {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    return NextResponse.json(
      {
        detail: error instanceof Error ? error.message : "Unable to reach MDNAC backend.",
        target: target.toString(),
      },
      { status: 502 },
    );
  }
}

function normalizedBaseUrl() {
  const value =
    process.env.MDNAC_API_URL ||
    process.env.NEXT_PUBLIC_MDNAC_API_URL ||
    "http://127.0.0.1:8000";

  return value.endsWith("/") ? value : `${value}/`;
}
