# training/hub_publisher.py
from __future__ import annotations
import os
import json
import torch


def create_model_card(
    model_name: str,
    architecture: str,
    description: str,
    metrics: dict,
    training_config: dict | None = None,
) -> str:
    metrics_table = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    config_section = ""
    if training_config:
        config_section = f"\n### Training Configuration\n```json\n{json.dumps(training_config, indent=2)}\n```\n"

    return f"""---
tags:
  - sequence-game
  - board-game
  - reinforcement-learning
  - imperfect-information
library_name: pytorch
---

# {model_name}

{description}

## Model Description

- **Architecture:** {architecture}
- **Game:** Sequence (10x10 board, card-based strategy game)
- **Type:** Policy-Value network for imperfect information game

## Training

Trained via self-play with AlphaZero-style reinforcement learning.
{config_section}

## Metrics

| Metric | Value |
|--------|-------|
{metrics_table}

## Usage

```python
import torch
from models.vit_policy_value import ViTPolicyValue

model = ViTPolicyValue()
model.load_state_dict(torch.load("model.pt", weights_only=True))
```

## Citation

```bibtex
@misc{{sequence-ai,
  title={{{model_name}}},
  year={{2026}},
}}
```
"""


def publish_model(
    model: torch.nn.Module,
    model_name: str,
    repo_id: str,
    card_content: str,
    token: str | None = None,
) -> str:
    save_dir = f"/tmp/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(card_content)
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=save_dir, repo_id=repo_id)
        return f"https://huggingface.co/{repo_id}"
    except ImportError:
        return save_dir
