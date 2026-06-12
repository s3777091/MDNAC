# MDNAC Agent Console

Next.js dashboard for the unified MDNAC backend.

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Environment

```bash
MDNAC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_MDNAC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_MDNAC_WS_URL=ws://127.0.0.1:8000/protein-span-completion/ws
```

`MDNAC_API_URL` is server-only and should point to the Docker service URL when running in Compose,
for example `http://protein-api:8000`. REST calls are proxied through `/api/backend/*`.

## Backend Contracts

- `GET /health`
- `POST /agent/run`
- `POST /agent/approve`
- `POST /agent/reject`
- `WS /protein-span-completion/ws`

## Docker

From the repository root:

```bash
docker compose -f docker/docker-compose.yaml up --build next-dashboard
```
