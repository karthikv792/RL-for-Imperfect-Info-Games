export interface CellData {
  card: string;
  occupant: number;
}

export interface LegalMove {
  position: number[];
  type: string;
}

export interface AgentMove {
  action: { player: number; position: number[]; type: string };
  thinking_time_ms: number;
}

export interface GameState {
  board: CellData[][];
  hand: string[];
  legal_moves: LegalMove[];
  turn: "human" | "ai";
  current_player: number;
  sequences: Record<string, number[][][]>;
  deck_size: number;
  is_over: boolean;
  winner: number | null;
  move_history: Array<{ player: number; position: number[]; type: string }>;
}
