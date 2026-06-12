import { Demo } from "@/components/demo";

export default function Home() {
  return (
    <main className="flex min-h-[100dvh] items-center justify-center overflow-x-hidden bg-black px-[clamp(1rem,4vw,4rem)] py-[clamp(1.5rem,6vh,5rem)] text-white">
      <Demo />
    </main>
  );
}
