"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const AGENTS = [
  { id: "random", name: "Random", description: "Makes random moves", difficulty: "Easy" },
  { id: "heuristic", name: "Heuristic", description: "Uses strategic rules", difficulty: "Medium" },
];

export default function Home() {
  const [selectedAgent, setSelectedAgent] = useState("heuristic");
  const router = useRouter();

  return (
    <main className="flex flex-col items-center justify-center min-h-screen px-4">
      <div className="text-center max-w-2xl">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-amber-600 bg-clip-text text-transparent">
          Sequence AI
        </h1>
        <p className="text-xl text-[var(--text-secondary)] mb-12">
          Challenge state-of-the-art AI agents at the classic card & board game
        </p>

        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-4 text-[var(--text-secondary)]">Choose Your Opponent</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-md mx-auto">
            {AGENTS.map((agent) => (
              <button
                key={agent.id}
                onClick={() => setSelectedAgent(agent.id)}
                className={`p-4 rounded-xl border-2 transition-all duration-200 text-left
                  ${selectedAgent === agent.id
                    ? "border-amber-500 bg-amber-500/10"
                    : "border-gray-700 bg-[var(--bg-card)] hover:border-gray-500"
                  }`}
              >
                <div className="font-semibold text-[var(--text-primary)]">{agent.name}</div>
                <div className="text-sm text-[var(--text-muted)]">{agent.description}</div>
                <div className={`text-xs mt-2 px-2 py-0.5 rounded-full inline-block
                  ${agent.difficulty === "Easy" ? "bg-green-500/20 text-green-400" : "bg-amber-500/20 text-amber-400"}
                `}>
                  {agent.difficulty}
                </div>
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={() => router.push(`/play?agent=${selectedAgent}`)}
          className="px-8 py-3 bg-gradient-to-r from-amber-500 to-amber-600 text-white font-semibold rounded-xl
            hover:from-amber-400 hover:to-amber-500 transition-all duration-200 shadow-lg shadow-amber-500/20
            hover:shadow-amber-500/30 active:scale-95"
        >
          Start Playing
        </button>

        <div className="mt-8">
          <button
            onClick={() => router.push("/spectate")}
            className="text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors text-sm"
          >
            or watch AI vs AI →
          </button>
        </div>
      </div>
    </main>
  );
}
