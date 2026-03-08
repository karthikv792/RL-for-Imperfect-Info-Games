# Sequence AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/karthikv792/sequence_rl/actions/workflows/ci.yml/badge.svg)](https://github.com/karthikv792/sequence_rl/actions)

State-of-the-art AI agents for the Sequence board game with a modern web interface.

## Features

- **Four advanced AI agents** combining tree search, reinforcement learning, and transformer architectures
- **Full game engine** implementing official 2-player Sequence rules with card management, sequence detection, and board logic
- **Self-play training pipeline** with experience buffers, Elo rating, arena evaluation, and HuggingFace Hub publishing
- **Modern web application** built with React/Next.js 15 and Tailwind CSS for playing against AI agents in the browser
- **FastAPI backend** with REST endpoints, WebSocket support for real-time play, and spectator mode for watching AI-vs-AI matches
- **Game replay system** with SQLite-backed storage for reviewing past games
- **Leaderboard and Elo tracking** to compare agent performance across matches
- **Model registry** for managing multiple trained agent checkpoints
- **Distributed training** support via HuggingFace Accelerate
- **Production deployment** with multi-stage Docker builds, Kubernetes manifests, and horizontal pod autoscaling
- **Comprehensive test suite** covering engine, agents, models, training, and API layers
- **CI/CD pipeline** with GitHub Actions for linting, testing, and Docker image builds

## Architecture

```
+------------------+     +-------------------+     +-------------------+
|                  |     |                   |     |                   |
|   Game Engine    +---->+    AI Agents      +---->+  Training Loop    |
|                  |     |                   |     |                   |
|  - game_state    |     |  - ISMCTS + ViT   |     |  - self_play      |
|  - actions       |     |  - ReBeL+DeBERTa  |     |  - trainer        |
|  - board         |     |  - Decision TF    |     |  - arena          |
|  - deck          |     |  - AlphaZero+     |     |  - elo            |
|  - rules         |     |    Belief         |     |  - exp. buffer    |
|                  |     |                   |     |                   |
+--------+---------+     +--------+----------+     +-------------------+
         |                        |
         |                        |
         v                        v
+------------------+     +-------------------+     +-------------------+
|                  |     |                   |     |                   |
|   Models (HF)    |     |   FastAPI (API)   +---->+   React/Next.js   |
|                  |     |                   |     |   Web App         |
|  - ViT policy/   |     |  - REST routes    |     |                   |
|    value         |     |  - WebSocket      |     |  - /play          |
|  - DeBERTa       |     |  - spectator      |     |  - /spectate      |
|    policy/value  |     |  - leaderboard    |     |  - /leaderboard   |
|  - Decision TF   |     |  - model registry |     |                   |
|  - board encoder |     |  - replay store   |     |                   |
|                  |     |                   |     |                   |
+------------------+     +-------------------+     +-------------------+
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/karthikv792/sequence_rl.git
cd sequence_rl
```

### 2. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install the package with all dependencies
pip install -e ".[dev]"
```

### 3. Run the API server

```bash
uvicorn api.main:app --reload --port 8000
```

The API documentation is available at `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc).

## Project Structure

```
sequence_rl/
├── engine/                     # Game engine
│   ├── game_state.py           # Core game state and information sets
│   ├── actions.py              # Action definitions (place, remove)
│   ├── board.py                # 10x10 board representation
│   ├── deck.py                 # Card deck management
│   └── rules.py                # Sequence detection and win conditions
├── agents/                     # AI agent implementations
│   ├── base.py                 # Abstract Agent base class
│   ├── random_agent.py         # Random baseline agent
│   ├── heuristic_agent.py      # Rule-based heuristic agent
│   ├── ismcts/                 # Information Set MCTS + ViT
│   │   ├── agent.py
│   │   ├── ismcts.py           # IS-MCTS algorithm
│   │   └── mcts.py             # Base MCTS implementation
│   ├── rebel/                  # ReBeL + DeBERTa
│   │   ├── agent.py
│   │   └── cfr.py              # Counterfactual regret minimization
│   ├── decision_tf/            # Decision Transformer
│   │   └── agent.py
│   └── alphazero_belief/       # AlphaZero + Belief State
│       └── agent.py
├── models/                     # Neural network architectures
│   ├── vit_policy_value.py     # ViT-based policy/value network
│   ├── deberta_policy_value.py # DeBERTa-based policy/value network
│   ├── decision_transformer.py # Decision Transformer model
│   ├── board_encoder.py        # Board state to image tensor encoder
│   └── board_tokenizer.py      # Board state to token sequence encoder
├── training/                   # Training pipeline
│   ├── train.py                # Main training entry point
│   ├── self_play.py            # Self-play game generation
│   ├── trainer.py              # Training loop and optimizer logic
│   ├── accelerate_trainer.py   # HuggingFace Accelerate integration
│   ├── arena.py                # Agent vs agent evaluation
│   ├── elo.py                  # Elo rating calculations
│   ├── experience_buffer.py    # Replay buffer for training examples
│   ├── hub_publisher.py        # HuggingFace Hub model publishing
│   └── configs/                # YAML training configurations
│       ├── ismcts_vit.yaml
│       ├── rebel_deberta.yaml
│       ├── decision_tf.yaml
│       └── alphazero_belief.yaml
├── api/                        # FastAPI backend
│   ├── main.py                 # Application entry point
│   ├── routes/
│   │   └── game.py             # REST API route handlers
│   ├── game_session.py         # Game session management
│   ├── ws_handler.py           # WebSocket handler for real-time play
│   ├── spectator.py            # AI-vs-AI spectator mode
│   ├── leaderboard.py          # Elo leaderboard management
│   ├── model_registry.py       # Trained model checkpoint registry
│   └── replay_store.py         # SQLite game replay storage
├── webapp/                     # React/Next.js frontend
│   ├── src/app/
│   │   ├── page.tsx            # Home page
│   │   ├── play/               # Play against AI
│   │   ├── spectate/           # Watch AI-vs-AI matches
│   │   └── leaderboard/        # Agent Elo rankings
│   └── package.json
├── tests/                      # Test suite
│   ├── test_engine/            # Engine unit tests
│   ├── test_agents/            # Agent unit tests
│   ├── test_models/            # Model unit tests
│   ├── test_training/          # Training pipeline tests
│   ├── test_api/               # API integration tests
│   └── test_benchmarks/        # Performance benchmarks
├── k8s/                        # Kubernetes manifests
│   ├── api-deployment.yaml     # API deployment, service, and HPA
│   ├── training-job.yaml       # Training job
│   ├── self-play-job.yaml      # Self-play generation job
│   ├── arena-job.yaml          # Arena evaluation job
│   ├── configmap.yaml          # Shared configuration
│   └── pvc.yaml                # Persistent volume claim for checkpoints
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── Dockerfile                  # Multi-stage Docker build
├── pyproject.toml              # Python project configuration
└── accelerate_config.yaml      # HuggingFace Accelerate config
```

## AI Agents

| Agent | Architecture | Description |
|-------|-------------|-------------|
| **ISMCTS + ViT** | Information Set MCTS with Vision Transformer | Uses determinization-based MCTS to handle hidden information, guided by a ViT (`google/vit-base-patch16-224`) policy/value network that processes the board as an image-like tensor. |
| **ReBeL + DeBERTa** | Recursive Belief Learning with DeBERTa | Combines counterfactual regret minimization (CFR) with a DeBERTa language model for policy and value estimation over tokenized board states. |
| **Decision Transformer** | Offline RL via sequence modeling | Frames gameplay as a sequence modeling problem, conditioning on desired returns to generate optimal actions from offline game trajectories. |
| **AlphaZero + Belief** | AlphaZero with belief state tracking | Extends the AlphaZero self-play paradigm with explicit belief tracking over opponent hidden information (hand cards), maintaining probability distributions over possible opponent states. |

Two baseline agents are also included for benchmarking:
- **Random Agent** -- selects uniformly from legal actions
- **Heuristic Agent** -- uses hand-crafted rules and positional evaluation

## Training

### Running Training

Train the default ISMCTS + ViT agent:

```bash
python -m training.train --config training/configs/ismcts_vit.yaml --iterations 100
```

Train a specific agent by pointing to its configuration:

```bash
# ReBeL + DeBERTa
python -m training.train --config training/configs/rebel_deberta.yaml --iterations 100

# Decision Transformer
python -m training.train --config training/configs/decision_tf.yaml --iterations 100

# AlphaZero + Belief
python -m training.train --config training/configs/alphazero_belief.yaml --iterations 100
```

### Training with HuggingFace Accelerate

For distributed or mixed-precision training:

```bash
accelerate launch --config_file accelerate_config.yaml -m training.train \
    --config training/configs/ismcts_vit.yaml \
    --iterations 200
```

### Configuration

Training configurations are stored as YAML files in `training/configs/`. Each config specifies:

- **model** -- backbone architecture, hidden layer sizes, input encoding parameters
- **mcts** -- simulation count, exploration constant (cPUCT), determinization count, Dirichlet noise
- **training** -- batch size, learning rate, weight decay, epochs per iteration, buffer size
- **self_play** -- games per iteration, parallel workers, max moves per game
- **arena** -- evaluation games, promotion threshold for accepting new models
- **monitoring** -- Weights & Biases project name, logging frequency

### Arena Evaluation

After training, evaluate agents against each other in the arena:

```bash
python -m training.arena --agent1 ismcts --agent2 rebel --games 100
```

### Monitoring

Training metrics are logged to [Weights & Biases](https://wandb.ai/) under the project name specified in the config (default: `sequence-ai`).

## Web Application

### Starting the Backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Starting the Frontend

```bash
cd webapp
npm install
npm run dev
```

The frontend runs at `http://localhost:3000` and connects to the API at `http://localhost:8000`.

### Pages

| Route | Description |
|-------|-------------|
| `/` | Home page |
| `/play` | Play a game against an AI agent |
| `/spectate` | Watch two AI agents play against each other in real time |
| `/leaderboard` | View agent Elo ratings and match history |

## API Reference

The API is served at `http://localhost:8000`. Full interactive documentation is available at `/docs` (Swagger UI).

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/agents` | List all available AI agents |
| `POST` | `/api/game` | Create a new game session against an AI agent |
| `GET` | `/api/game/{session_id}` | Get current game state |
| `POST` | `/api/game/{session_id}/move` | Make a move (place or remove a chip) |
| `POST` | `/api/spectate` | Start an AI-vs-AI spectator match |
| `POST` | `/api/spectate/{match_id}/advance` | Advance a spectator match by one step |
| `GET` | `/api/spectate` | List active spectator matches |
| `GET` | `/api/leaderboard` | Get agent Elo leaderboard |
| `GET` | `/api/models` | List registered model checkpoints |
| `WS` | `/ws` | WebSocket endpoint for real-time game interaction |

### Example: Create and Play a Game

```bash
# Create a game against the ISMCTS agent
curl -X POST http://localhost:8000/api/game \
  -H "Content-Type: application/json" \
  -d '{"agent": "ismcts", "human_player": 1}'

# Make a move (place a chip at position [3, 4])
curl -X POST http://localhost:8000/api/game/{session_id}/move \
  -H "Content-Type: application/json" \
  -d '{"position": [3, 4], "type": "place"}'
```

## Deployment

### Docker

Build and run the API server in a container:

```bash
# Build the image
docker build -t sequence-ai .

# Run the container
docker run -p 8000:8000 sequence-ai
```

The Docker image uses a multi-stage build (builder + runtime) to minimize image size. The runtime stage only includes the Python virtual environment and application source code.

### Kubernetes

Deploy to a Kubernetes cluster:

```bash
# Apply all manifests
kubectl apply -f k8s/

# This deploys:
#   - API server (2 replicas with HPA scaling up to 10)
#   - Training job
#   - Self-play job
#   - Arena evaluation job
#   - ConfigMap for shared settings
#   - PersistentVolumeClaim for model checkpoints
```

The API deployment includes liveness and readiness probes, resource requests/limits, and a HorizontalPodAutoscaler that scales based on CPU utilization (target: 70%).

## Testing

### Run the Full Test Suite

```bash
pytest tests/ -v --tb=short
```

### Run Tests by Category

```bash
# Engine tests
pytest tests/test_engine/ -v

# Agent tests
pytest tests/test_agents/ -v

# Model tests
pytest tests/test_models/ -v

# Training pipeline tests
pytest tests/test_training/ -v

# API tests
pytest tests/test_api/ -v

# Performance benchmarks
pytest tests/test_benchmarks/ -v
```

### Linting

```bash
ruff check .
```

### Code Coverage

```bash
pytest tests/ --cov=engine --cov=agents --cov=models --cov=training --cov=api --cov-report=term-missing
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make your changes and add tests
5. Run the test suite: `pytest tests/ -v`
6. Run the linter: `ruff check .`
7. Commit your changes and push to your fork
8. Open a pull request

## License

This project is licensed under the MIT License.
