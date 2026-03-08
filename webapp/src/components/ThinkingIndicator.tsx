"use client";

export function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-2 text-amber-400">
      <div className="flex gap-1">
        {[0, 1, 2].map(i => (
          <div
            key={i}
            className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-bounce"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
      <span className="text-sm">AI is thinking...</span>
    </div>
  );
}
