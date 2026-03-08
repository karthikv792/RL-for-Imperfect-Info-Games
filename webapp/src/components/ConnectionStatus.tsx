"use client";

interface ConnectionStatusProps {
  isConnected: boolean;
  reconnectAttempt: number;
}

export function ConnectionStatus({ isConnected, reconnectAttempt }: ConnectionStatusProps) {
  if (isConnected) return null;

  return (
    <div className="fixed top-0 left-0 right-0 bg-red-500/90 text-white text-center py-2 text-sm z-50">
      <span className="animate-pulse">
        {reconnectAttempt > 0
          ? `Reconnecting... (attempt ${reconnectAttempt})`
          : "Disconnected from server"}
      </span>
    </div>
  );
}
