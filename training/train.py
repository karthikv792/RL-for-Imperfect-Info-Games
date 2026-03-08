# training/train.py
"""Main training entry point for ISMCTS + ViT agent."""
from __future__ import annotations
import argparse
import os
import yaml
import torch

from models.vit_policy_value import ViTPolicyValue
from models.board_encoder import BoardEncoder
from training.trainer import SequenceTrainer, TrainingConfig
from training.self_play import run_self_play_game, SelfPlayConfig
from training.experience_buffer import ExperienceBuffer


def main():
    parser = argparse.ArgumentParser(description="Train Sequence AI agent")
    parser.add_argument("--config", type=str, default="training/configs/ismcts_vit.yaml")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Initialize model
    model_cfg = config["model"]
    model = ViTPolicyValue(
        backbone_name=model_cfg["backbone"],
        num_actions=300,
        policy_hidden=model_cfg["policy_hidden"],
        value_hidden=model_cfg["value_hidden"],
        num_input_channels=model_cfg["num_input_channels"],
    ).to(device)

    encoder = BoardEncoder(target_size=model_cfg["target_size"])

    # Training config
    training_config = TrainingConfig(
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        epochs_per_iteration=config["training"]["epochs_per_iteration"],
        target_size=model_cfg["target_size"],
    )
    trainer = SequenceTrainer(model=model, config=training_config, device=device)

    # Self-play config
    sp_cfg = config["self_play"]
    mcts_cfg = config["mcts"]
    self_play_config = SelfPlayConfig(
        num_simulations=mcts_cfg["simulations"],
        num_determinizations=mcts_cfg["determinizations"],
        cpuct=mcts_cfg["cpuct"],
        dirichlet_alpha=mcts_cfg["dirichlet_alpha"],
        dirichlet_epsilon=mcts_cfg["dirichlet_epsilon"],
        max_moves=sp_cfg["max_moves"],
    )

    buffer = ExperienceBuffer(max_size=config["training"]["buffer_size"])

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.iterations} ===")

        # Self-play phase
        print(f"Running {sp_cfg['games_per_iteration']} self-play games...")
        for game_idx in range(sp_cfg["games_per_iteration"]):
            examples = run_self_play_game(
                model=model,
                encoder=encoder,
                config=self_play_config,
                seed=iteration * 10000 + game_idx,
                device=device,
            )
            for state, policy, value in examples:
                buffer.add(state, policy, value)

        print(f"Buffer size: {len(buffer)}")

        # Training phase
        if len(buffer) >= training_config.batch_size:
            print("Training...")
            for epoch in range(training_config.epochs_per_iteration):
                states, policies, values = buffer.sample(
                    min(len(buffer), training_config.batch_size * 10)
                )
                metrics = trainer.train_epoch(states, policies, values)
                print(f"  Epoch {epoch + 1}: {metrics}")

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"iteration_{iteration + 1}")
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
