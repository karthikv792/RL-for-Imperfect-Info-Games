# Sequence AI Agent — Design Document

**Date:** 2026-03-01
**Status:** Approved
**Goal:** Build a professional-grade Sequence board game with state-of-the-art AI agents using HuggingFace Transformers, playable via a web interface.

---

## 1. Scope & Constraints

- **2-player only** (individual, no teams)
- **4 AI agent architectures** (2 built first, 2 added later):
  1. ISMCTS + ViT (image-based, phase 1)
  2. ReBeL-style CFR + DeBERTa (sequence-based, phase 1)
  3. Decision Transformer (phase 2)
  4. Classic AlphaZero + belief sampling (phase 2)
- **Web app** for human vs AI play, with AI vs AI spectator mode
- **Pretrained HF models** with custom policy/value heads
- **Training**: Local M4 Max (MPS) + Kubernetes for cloud scale
- **Distributed training** via HuggingFace Accelerate
- **Monitoring**: Weights & Biases integration

---

## 2. Project Structure (Monorepo)

```
sequence-ai/
├── engine/                 # Pure Python game logic (zero UI deps)
│   ├── __init__.py
│   ├── board.py            # Board representation and operations
│   ├── deck.py             # Card deck management
│   ├── game_state.py       # Immutable GameState dataclass
│   ├── actions.py          # Action types and encoding
│   ├── rules.py            # Win detection, legal move generation
│   └── information_set.py  # InformationSet for imperfect info
│
├── agents/                 # Agent framework
│   ├── __init__.py
│   ├── base.py             # Agent protocol (interface)
│   ├── random_agent.py     # Random baseline
│   ├── heuristic_agent.py  # Manhattan distance heuristic (from existing code)
│   ├── human_agent.py      # Human player adapter
│   ├── ismcts/             # ISMCTS + ViT agent
│   │   ├── __init__.py
│   │   ├── mcts.py         # MCTS tree search
│   │   ├── ismcts.py       # Information Set MCTS (determinization)
│   │   └── agent.py        # Agent wrapper
│   ├── rebel/              # ReBeL-style CFR + DeBERTa agent
│   │   ├── __init__.py
│   │   ├── cfr.py          # Counterfactual Regret Minimization
│   │   ├── belief.py       # Public belief state computation
│   │   └── agent.py        # Agent wrapper
│   ├── decision_tf/        # Decision Transformer (phase 2)
│   └── alphazero_belief/   # Classic AlphaZero + belief sampling (phase 2)
│
├── models/                 # HuggingFace model definitions
│   ├── __init__.py
│   ├── vit_policy_value.py # ViT backbone + policy/value heads
│   ├── deberta_policy_value.py  # DeBERTa backbone + policy/value heads
│   ├── board_tokenizer.py  # Custom tokenizer for board states (DeBERTa)
│   └── board_encoder.py    # Board → multi-channel tensor (ViT)
│
├── training/               # Training pipeline
│   ├── __init__.py
│   ├── self_play.py        # Self-play game generation
│   ├── trainer.py          # HF Trainer wrapper with custom data collator
│   ├── arena.py            # New model vs champion evaluation
│   ├── experience_buffer.py # Rolling training example buffer
│   ├── elo.py              # Elo rating system
│   ├── configs/            # YAML training configs
│   │   ├── ismcts_vit.yaml
│   │   └── rebel_deberta.yaml
│   └── train.py            # Main training entry point
│
├── webapp/                 # React/Next.js frontend
│   ├── src/
│   │   ├── components/     # Board, Hand, GameInfo, AgentSelector
│   │   ├── hooks/          # WebSocket, game state hooks
│   │   └── pages/          # Play, Spectate, Leaderboard
│   └── package.json
│
├── api/                    # FastAPI backend
│   ├── __init__.py
│   ├── main.py             # FastAPI app
│   ├── game_session.py     # Session management
│   ├── ws_handler.py       # WebSocket handler
│   └── routes/             # REST endpoints
│
├── k8s/                    # Kubernetes manifests
│   ├── training-job.yaml
│   ├── self-play-job.yaml
│   ├── arena-job.yaml
│   ├── api-deployment.yaml
│   ├── pvc.yaml            # Persistent volume for checkpoints
│   └── configmap.yaml
│
├── tests/
│   ├── test_engine/
│   ├── test_agents/
│   ├── test_models/
│   └── test_training/
│
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## 3. Game Engine

### Board Representation

10x10 grid. Each cell has a `card_type` (suit+rank) and `occupant` (empty/player1/player2/corner).

### Multi-Channel Tensor Encoding

For neural network input, the board is encoded as a multi-channel tensor:

| Channels | Description |
|----------|-------------|
| 0 | Player 1 tokens (binary 10x10) |
| 1 | Player 2 tokens (binary 10x10) |
| 2 | Corner/free spaces (binary 10x10) |
| 3-6 | Card suits on each cell (one-hot, 4 channels) |
| 7-19 | Card ranks on each cell (one-hot, 13 channels) |
| 20 | Current player's playable positions |
| 21 | Sequences already formed |

Total: 22 channels x 10x10.

### GameState Dataclass

```python
@dataclass(frozen=True)
class GameState:
    board: np.ndarray            # 10x10 occupancy grid
    board_cards: np.ndarray      # 10x10 card layout (static)
    current_player: int          # 1 or 2
    hands: dict[int, tuple[str, ...]]  # player_id -> cards (frozen)
    deck_size: int               # cards remaining in deck
    discard_pile: tuple[str, ...]  # visible discards
    sequences: dict[int, tuple]  # completed sequences per player
```

Immutable — `apply_action()` returns a new `GameState`.

### Key Interfaces

- `GameState.get_legal_actions() -> list[Action]`
- `GameState.apply_action(action: Action) -> GameState`
- `GameState.is_terminal() -> bool`
- `GameState.get_winner() -> Optional[int]`
- `GameState.to_tensor(player_perspective: int) -> np.ndarray` (22x10x10)
- `GameState.to_information_set(player: int) -> InformationSet`

### Action Space

300 discrete actions:
- 0-99: Place token at position (row*10+col) using matching card
- 100-199: Remove opponent token at position using one-eyed Jack (J1)
- 200-299: Place token at position using two-eyed Jack (J2, wild)

### InformationSet

What a player can observe (excludes opponent's hand and deck order):

```python
@dataclass(frozen=True)
class InformationSet:
    board: np.ndarray
    current_player: int
    own_hand: tuple[str, ...]
    deck_size: int
    discard_pile: tuple[str, ...]
    sequences: dict[int, tuple]
```

---

## 4. Agent Framework

### Agent Protocol

```python
class Agent(Protocol):
    name: str

    def select_action(
        self,
        state: GameState,
        info_set: InformationSet
    ) -> Action: ...

    def train(self, training_data: TrainingData) -> Metrics: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

All agents implement this interface. Human players, random agents, and AI agents all look the same to the game engine.

### Agent 1: ISMCTS + ViT (Image-Based)

**Algorithm: Information Set Monte Carlo Tree Search**

1. Receive `InformationSet` (own hand, board, discards — NOT opponent's hand)
2. **Determinize**: Sample N possible opponent hands consistent with observations (cards not in own hand, not on board, not in discard)
3. For each determinization, run standard MCTS using the ViT network for:
   - **Policy prior**: Guides which actions to explore first
   - **Value estimate**: Evaluates leaf nodes without full rollout
4. Aggregate action visit counts across all determinizations
5. Select action with highest aggregate visit count

**ViT Architecture:**

- **Backbone**: `google/vit-base-patch16-224` (pretrained on ImageNet)
- **Input**: Board tensor (22 channels x 10x10) upscaled to match ViT expectations
- **Custom heads** attached to ViT's `[CLS]` token output:
  - Policy head: `Linear(768, 512) → ReLU → Linear(512, 300) → Softmax`
  - Value head: `Linear(768, 256) → ReLU → Linear(256, 1) → Tanh`
- **Training**: Freeze backbone initially, train heads only. Then unfreeze and fine-tune full model at lower learning rate.

### Agent 2: ReBeL-style CFR + DeBERTa (Sequence-Based)

**Algorithm: Regularized Beliefs Learning**

1. Maintain a **public belief state** (probability distribution over possible game states given public info)
2. At each decision point, run **CFR iterations** using DeBERTa's value estimates for leaf evaluation
3. CFR converges toward Nash equilibrium strategy for the current subgame
4. DeBERTa is trained to predict the value of public belief states

**DeBERTa Architecture:**

- **Backbone**: `microsoft/deberta-v3-small` (pretrained)
- **Input tokenization**: Board flattened to 100 cell tokens, each encoded as `[suit_id, rank_id, occupant_id]`. Hand cards appended as additional tokens. Special tokens for deck size and discard info.
- **Custom heads**:
  - Policy head: Over final hidden states → `Linear(768, 300) → Softmax`
  - Value head: Over `[CLS]` → `Linear(768, 1) → Tanh`

### Baseline Agents (from existing code)

- **RandomAgent**: Uniform random legal move selection
- **HeuristicAgent**: Manhattan distance clustering (current BasePolicyAgent logic)

---

## 5. Training Pipeline

### Self-Play Engine

```
Self-Play Workers  →  Experience Buffer  →  HF Trainer  →  Arena  →  Champion Update
     ↑                                                         │
     └─────────────────────────────────────────────────────────┘
                         (new champion feeds back)
```

1. **Self-play**: N parallel games using current champion model with exploration noise (Dirichlet noise on root prior). Each game produces training examples: `(state_tensor, mcts_policy, game_outcome)`.
2. **Experience buffer**: Rolling window of recent examples (configurable, default 500K).
3. **Training**: HuggingFace Trainer with custom `DataCollator`. Joint loss = cross-entropy(policy) + MSE(value). Uses HF Accelerate for device placement.
4. **Arena**: New model plays 100 games against champion. Promoted if win rate > 55%.
5. **Checkpoint**: Models saved in HF format. Can push to HuggingFace Hub.

### Distributed Training (HF Accelerate)

- Training code uses `Accelerator` for device-agnostic execution
- Single script works on: M4 Max (MPS), single NVIDIA GPU, multi-GPU, multi-node
- Self-play can run on CPU workers while training runs on GPU
- `accelerate launch --config_file accelerate_config.yaml train.py`

### Training Configuration (YAML)

```yaml
# configs/ismcts_vit.yaml
model:
  backbone: "google/vit-base-patch16-224"
  freeze_backbone_epochs: 5
  policy_hidden: 512
  value_hidden: 256

mcts:
  simulations: 400
  cpuct: 1.5
  determinizations: 30
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs_per_iteration: 10
  buffer_size: 500000

self_play:
  games_per_iteration: 100
  parallel_workers: 8

arena:
  games: 100
  promotion_threshold: 0.55

monitoring:
  wandb_project: "sequence-ai"
  log_every_n_steps: 10
```

### M4 Max Optimizations

- MPS backend for PyTorch (`torch.device("mps")`)
- `torch.compile()` for model graph optimization
- Parallel self-play via `multiprocessing` (8+ performance cores)
- Mixed precision where MPS supports it

---

## 6. Web Application

### Frontend (React / Next.js)

**Components:**
- **BoardView**: 10x10 SVG grid. Card faces, token overlays, legal move highlights
- **HandView**: Player's cards as clickable card images
- **MovePreview**: When card selected, highlight valid board positions
- **AgentSelector**: Dropdown to choose opponent agent + difficulty
- **GameInfo**: Turn indicator, sequences formed, deck count, discard pile
- **MoveHistory**: Scrollable log of moves
- **SpectatorView**: Watch AI vs AI games live, with board heatmap of agent's policy distribution

**Pages:**
- `/play` — Human vs AI game
- `/spectate` — Watch live AI vs AI matches
- `/leaderboard` — Elo ratings, head-to-head stats

### Backend (FastAPI)

**WebSocket Protocol:**

```
Client → Server:
  {"type": "new_game", "agent": "ismcts_vit", "difficulty": "hard"}
  {"type": "make_move", "action": {"position": [3, 5], "type": "place"}}
  {"type": "spectate", "match_id": "abc123"}

Server → Client:
  {"type": "game_state", "board": [...], "hand": [...], "legal_moves": [...]}
  {"type": "agent_move", "action": {...}, "thinking_time_ms": 1200}
  {"type": "game_over", "winner": 1, "sequences": [...]}
  {"type": "spectate_update", "board": [...], "last_move": {...}, "policy_heatmap": [...]}
```

**REST Endpoints:**
- `GET /api/agents` — List available agents
- `GET /api/leaderboard` — Elo ratings
- `GET /api/game/{id}/replay` — Game replay data

### Spectator Mode

- AI vs AI games run continuously on the server
- Spectators connect via WebSocket and receive board updates in real-time
- Optional policy heatmap overlay showing what the agent "thinks"
- Multiple concurrent AI vs AI matches viewable

---

## 7. Monitoring & Logging

### Training Monitoring (Weights & Biases)

Integrated via HuggingFace Trainer's `WandbCallback`:
- Policy loss, value loss, total loss per step
- Self-play game length distribution
- Arena win rates (new challenger vs champion)
- Elo rating progression over training iterations
- Learning rate schedule
- Device utilization (MPS/GPU)
- MCTS statistics: average tree depth, nodes explored, time per search

### Game Logging

- Every game saved as JSON (replayable):
  - Full move history with timestamps
  - Board state at each step
  - Agent's policy distribution per move
  - Value estimates over the game
- Stored in `data/replays/` directory

### Web App Monitoring

- FastAPI middleware for request/response logging
- WebSocket connection stats (active sessions, disconnects)
- Agent inference latency tracking

---

## 8. Testing & Evaluation

### Test Strategy

| Layer | What | Tools |
|-------|------|-------|
| Unit | Board ops, win detection, legal moves, deck, actions | pytest |
| Integration | Agent produces valid moves, training loop converges on toy game | pytest |
| End-to-end | Full game completes without error, web app serves correctly | pytest + playwright |
| Performance | Moves/sec, MCTS rollouts/sec, training throughput | pytest-benchmark |

### Agent Evaluation

- **Elo rating system**: Round-robin tournaments between all agents
- **Win rate matrix**: Head-to-head records for every agent pair
- **Strength progression**: Elo over training iterations (learning curves)
- **Baseline calibration**: Random and Heuristic agents as fixed reference points

---

## 9. Kubernetes Deployment

### Training Jobs

```yaml
# Self-play → Training → Arena pipeline
self-play-job:
  image: sequence-ai:latest
  command: ["python", "-m", "training.self_play"]
  resources: {cpu: "4", memory: "8Gi"}

training-job:
  image: sequence-ai:latest
  command: ["accelerate", "launch", "training/train.py"]
  resources: {cpu: "4", memory: "16Gi", gpu: "1"}

arena-job:
  image: sequence-ai:latest
  command: ["python", "-m", "training.arena"]
  resources: {cpu: "4", memory: "8Gi"}
```

### Infrastructure

- **PersistentVolumeClaim**: Shared storage for experience buffer + model checkpoints
- **ConfigMap**: Training YAML configs injected as environment
- **CronJob** (optional): Automated training iterations on schedule
- **API Deployment**: FastAPI server with horizontal pod autoscaling

---

## 10. Phase Plan

**Phase 1** (parallel tracks):
- Track A: Game engine rewrite (clean, immutable, tested)
- Track B: ISMCTS + ViT agent (MCTS + pretrained ViT + custom heads)

**Phase 2:**
- ReBeL + DeBERTa agent
- Web application (frontend + backend)
- Training pipeline with Accelerate

**Phase 3:**
- Decision Transformer agent
- Classic AlphaZero + belief sampling agent
- Kubernetes manifests
- Elo leaderboard and spectator mode

**Phase 4:**
- Polish, optimization, deployment
- HuggingFace Hub model publishing
- Documentation
