"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface AgentStats {
  agent: string;
  elo: number;
  wins: number;
  losses: number;
  draws: number;
  games: number;
}

export default function LeaderboardPage() {
  const [stats, setStats] = useState<AgentStats[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const apiUrl = typeof window !== "undefined"
      ? `http://${window.location.hostname}:8000/api/leaderboard`
      : "http://localhost:8000/api/leaderboard";

    fetch(apiUrl)
      .then(res => res.json())
      .then(data => {
        setStats(data.leaderboard || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <main className="flex flex-col items-center min-h-screen px-4 py-8">
      <div className="w-full max-w-2xl">
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={() => router.push("/")}
            className="text-sm text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          >
            ← Back
          </button>
          <h1 className="text-3xl font-bold text-[var(--text-primary)]">Leaderboard</h1>
          <div className="w-12" />
        </div>

        {loading ? (
          <div className="text-center text-[var(--text-muted)] py-12">Loading...</div>
        ) : stats.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-[var(--text-muted)] mb-2">No games played yet</p>
            <p className="text-sm text-[var(--text-muted)]">Play some games to populate the leaderboard</p>
          </div>
        ) : (
          <div className="bg-[var(--bg-card)] rounded-xl overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-sm text-[var(--text-muted)] font-medium">Rank</th>
                  <th className="text-left py-3 px-4 text-sm text-[var(--text-muted)] font-medium">Agent</th>
                  <th className="text-right py-3 px-4 text-sm text-[var(--text-muted)] font-medium">Elo</th>
                  <th className="text-right py-3 px-4 text-sm text-[var(--text-muted)] font-medium">W</th>
                  <th className="text-right py-3 px-4 text-sm text-[var(--text-muted)] font-medium">L</th>
                  <th className="text-right py-3 px-4 text-sm text-[var(--text-muted)] font-medium">D</th>
                  <th className="text-right py-3 px-4 text-sm text-[var(--text-muted)] font-medium">Games</th>
                </tr>
              </thead>
              <tbody>
                {stats.map((agent, i) => (
                  <tr key={agent.agent} className="border-b border-gray-800 last:border-0 hover:bg-white/5 transition-colors">
                    <td className="py-3 px-4 text-sm text-[var(--text-secondary)]">
                      {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : `#${i + 1}`}
                    </td>
                    <td className="py-3 px-4 text-sm font-medium text-[var(--text-primary)] capitalize">{agent.agent}</td>
                    <td className="py-3 px-4 text-sm text-right text-amber-400 font-mono">{Math.round(agent.elo)}</td>
                    <td className="py-3 px-4 text-sm text-right text-green-400">{agent.wins}</td>
                    <td className="py-3 px-4 text-sm text-right text-red-400">{agent.losses}</td>
                    <td className="py-3 px-4 text-sm text-right text-[var(--text-muted)]">{agent.draws}</td>
                    <td className="py-3 px-4 text-sm text-right text-[var(--text-secondary)]">{agent.games}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}
