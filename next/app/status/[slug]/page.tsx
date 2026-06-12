import { ArrowLeft, CheckCircle2, Loader2, PauseCircle, Radio, ShieldAlert } from "lucide-react";
import Link from "next/link";

import { STATUS_SCREENS, statusBySlug } from "@/lib/mdnac";
import { cn } from "@/lib/utils";

export function generateStaticParams() {
  return STATUS_SCREENS.map((screen) => ({ slug: screen.slug }));
}

export default async function StatusPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const screen =
    statusBySlug(slug) ??
    ({
      slug,
      event: slug.replaceAll("-", "_"),
      label: slug.replaceAll("-", " "),
      phase: "Custom",
      tone: "neutral" as const,
      description: "This status was emitted locally or is not in the backend reference map yet.",
      nextAction: "Return to the input page and inspect the raw event payload.",
    });

  return (
    <main className="min-h-[100dvh] bg-white px-4 py-6 text-black sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-5xl flex-col gap-5">
        <Link
          href="/"
          className="inline-flex w-fit items-center gap-2 rounded-full border border-black/10 bg-black/5 px-4 py-2 text-sm font-semibold text-black transition hover:bg-black/10"
        >
          <ArrowLeft className="size-4" />
          Back to input
        </Link>

        <section className="rounded-3xl border border-black/10 bg-white p-6 shadow-sm md:p-8">
          <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="mb-4 flex items-center gap-3">
                <span
                  className={cn(
                    "grid size-12 place-items-center rounded-2xl",
                    toneClass(screen.tone),
                  )}
                >
                  {toneIcon(screen.tone)}
                </span>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-black/50">
                    {screen.phase}
                  </p>
                  <h1 className="text-3xl font-semibold capitalize tracking-tight text-black md:text-5xl">
                    {screen.label}
                  </h1>
                </div>
              </div>
              <p className="max-w-3xl text-lg leading-8 text-black/70">{screen.description}</p>
            </div>

            <div className="rounded-2xl border border-black/10 bg-black/5 px-4 py-3">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-black/50">
                Event
              </p>
              <code className="mt-1 block font-mono text-sm font-semibold text-black">
                {screen.event}
              </code>
            </div>
          </div>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <InfoTile label="Route" value="/protein-span-completion/ws" />
            <InfoTile label="REST fallback" value="/agent/run" />
            <InfoTile label="Navigation" value={`/status/${screen.slug}`} />
          </div>

          <div className="mt-8 rounded-3xl border border-black/10 bg-black/5 p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-black/50">
              Next action
            </p>
            <p className="mt-2 text-base leading-7 text-black/70">{screen.nextAction}</p>
          </div>
        </section>
      </div>
    </main>
  );
}

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-3xl border border-black/10 bg-black/5 p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-black/50">{label}</p>
      <p className="mt-2 break-all font-mono text-sm text-black/70">{value}</p>
    </div>
  );
}

function toneClass(tone: string) {
  if (tone === "success") {
    return "border border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  if (tone === "active") {
    return "border border-cyan-200 bg-cyan-50 text-cyan-700";
  }
  if (tone === "warning") {
    return "border border-amber-200 bg-amber-50 text-amber-700";
  }
  if (tone === "danger") {
    return "border border-rose-200 bg-rose-50 text-rose-700";
  }
  return "border border-black/10 bg-black/5 text-black/60";
}

function toneIcon(tone: string) {
  if (tone === "success") {
    return <CheckCircle2 className="size-5" />;
  }
  if (tone === "active") {
    return <Loader2 className="size-5 animate-spin" />;
  }
  if (tone === "warning") {
    return <PauseCircle className="size-5" />;
  }
  if (tone === "danger") {
    return <ShieldAlert className="size-5" />;
  }
  return <Radio className="size-5" />;
}
