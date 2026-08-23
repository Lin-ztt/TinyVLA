#!/usr/bin/env python3
"""Run and resume frozen-SmolVLA EXPO training on LIBERO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import yaml
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.utils.random_utils import set_seed

from tinyvla.expo import EXPOConfig, EXPOLearner, EXPOReplayBuffer
from tinyvla.expo.rollout import EXPORolloutConfig, rollout_episode
from tinyvla.libero import make_libero_config, make_smolvla_stack


CHECKPOINT_FORMAT_VERSION = 3
ENTROPY_COORDINATE = "residual_unscaled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/expo/train.yaml"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--verify-resume",
        action="store_true",
        help="After loading a completed checkpoint, force one more episode and update.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as file:
        config = yaml.safe_load(file)
    training = config["training"]
    batch_size = int(config["expo"]["batch_size"])
    if int(training["learning_starts"]) < batch_size:
        raise ValueError("learning_starts must be at least batch_size")
    for name in (
        "smoke_utd_ratio",
        "smoke_gradient_steps",
        "utd_ratio",
        "min_episodes",
        "min_chunk_transitions",
        "min_gradient_steps",
        "save_every_episodes",
    ):
        if int(training[name]) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if not isinstance(training.get("warmup_base_only", False), bool):
        raise ValueError("training.warmup_base_only must be a boolean")
    if not 0.0 <= float(training.get("base_exploration_prob", 0.0)) <= 1.0:
        raise ValueError("training.base_exploration_prob must be in [0, 1]")
    if int(training.get("save_every_gradient_steps", 1)) <= 0:
        raise ValueError("training.save_every_gradient_steps must be positive")
    if int(training.get("critic_warmup_gradient_steps", 0)) < 0:
        raise ValueError("training.critic_warmup_gradient_steps must be non-negative")
    environment = config["environment"]
    if "train_init_state_count" in environment:
        if int(environment["train_init_state_count"]) <= 0:
            raise ValueError("environment.train_init_state_count must be positive")
        if int(environment.get("train_init_state_start", 0)) < 0:
            raise ValueError("environment.train_init_state_start must be non-negative")
        if not isinstance(environment.get("shuffle_init_states", False), bool):
            raise ValueError("environment.shuffle_init_states must be a boolean")
    return config


def training_init_state_id(
    episode_index: int, seed: int, environment: dict[str, Any]
) -> int:
    if "train_init_state_count" not in environment:
        return episode_index
    start = int(environment.get("train_init_state_start", 0))
    count = int(environment["train_init_state_count"])
    cycle, offset = divmod(episode_index, count)
    if not environment.get("shuffle_init_states", False):
        return start + offset
    order = np.random.default_rng(seed + cycle).permutation(count)
    return start + int(order[offset])


def parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def learner_fingerprints(learner: EXPOLearner) -> dict[str, str]:
    temperature = learner.log_temperature.detach().cpu().contiguous()
    return {
        "actor": parameter_hash(learner.actor),
        "critic": parameter_hash(learner.critic),
        "target_critic": parameter_hash(learner.target_critic),
        "image_encoder": parameter_hash(learner.image_encoder),
        "temperature": hashlib.sha256(temperature.numpy().tobytes()).hexdigest(),
    }


def make_expo_config(config: dict[str, Any]) -> EXPOConfig:
    expo = config["expo"]
    return EXPOConfig(
        actor_lr=float(expo["actor_lr"]),
        critic_lr=float(expo["critic_lr"]),
        temperature_lr=float(expo["temperature_lr"]),
        tau=float(expo["tau"]),
        edit_scale=float(expo["edit_scale"]),
        initial_temperature=float(expo["initial_temperature"]),
        target_entropy=float(expo["target_entropy"]),
        num_qs=int(expo["num_qs"]),
        num_min_qs=int(expo["num_min_qs"]),
        crop_padding=int(expo["crop_padding"]),
        rotation_degrees=float(expo["rotation_degrees"]),
        color_jitter=float(expo["color_jitter"]),
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def assert_finite(learner: EXPOLearner, metrics: dict[str, float] | None = None) -> None:
    if metrics is not None:
        invalid = {name: value for name, value in metrics.items() if not math.isfinite(value)}
        if invalid:
            raise FloatingPointError(f"Non-finite EXPO metrics: {invalid}")
    for name, parameter in learner.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite EXPO parameter: {name}")


def validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    for name in ("seed", "checkpoint", "device", "environment", "expo", "replay_buffer"):
        if saved[name] != current[name]:
            raise ValueError(f"Resume config mismatch: {name}")
    for name in (
        "learning_starts",
        "smoke_utd_ratio",
        "utd_ratio",
        "critic_warmup_gradient_steps",
    ):
        if saved["training"].get(name, 0) != current["training"].get(name, 0):
            raise ValueError(f"Resume config mismatch: training.{name}")
    if bool(saved["training"].get("warmup_base_only", False)) != bool(
        current["training"].get("warmup_base_only", False)
    ):
        raise ValueError("Resume config mismatch: training.warmup_base_only")


def validate_checkpoint_format(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported EXPO training checkpoint format")
    if checkpoint.get("entropy_coordinate") != ENTROPY_COORDINATE:
        raise ValueError("EXPO checkpoint uses an incompatible entropy coordinate")


def _validation_state(
    learner: EXPOLearner, buffer_state: dict[str, Any]
) -> dict[str, Any] | None:
    if int(buffer_state["size"]) == 0:
        return None
    pixels = torch.from_numpy(buffer_state["observation_pixels"][0:1]).to(learner.device)
    state = torch.from_numpy(buffer_state["observation_state"][0:1]).to(learner.device)
    executed_action = torch.from_numpy(buffer_state["executed_actions"][0:1]).to(
        learner.device
    )
    with torch.no_grad():
        residual = learner.act_residual(
            pixels, state, executed_action, deterministic=True
        )
        embedding = learner.inference_embedding(pixels)
        target_q = learner.target_critic(
            embedding, state, executed_action.reshape(1, 56)
        )
    probe = EXPOReplayBuffer.from_checkpoint_state(buffer_state)
    return {
        "pixels": buffer_state["observation_pixels"][0].copy(),
        "state": buffer_state["observation_state"][0].copy(),
        "executed_action": buffer_state["executed_actions"][0].copy(),
        "base_action": buffer_state["base_actions"][0].copy(),
        "deterministic_residual": residual[0].cpu().numpy(),
        "target_q": target_q[0].cpu().numpy(),
        "temperature": float(learner.temperature.detach().cpu()),
        "next_buffer_batch": probe.sample(min(4, len(probe))),
    }


def save_training_state(
    output_dir: Path,
    learner: EXPOLearner,
    buffer: EXPOReplayBuffer,
    counters: dict[str, int],
    episode_index: int,
    episodes: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    config: dict[str, Any],
    policy_hash: str,
    initial_fingerprints: dict[str, str],
    checkpoint_name: str = "checkpoint_latest.pt",
) -> None:
    buffer_state = buffer.checkpoint_state()
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "entropy_coordinate": ENTROPY_COORDINATE,
        "config": config,
        "smolvla_checkpoint": str(Path(config["checkpoint"]).resolve()),
        "smolvla_parameter_hash": policy_hash,
        "initial_learner_fingerprints": initial_fingerprints,
        "expo": learner.checkpoint_state(metadata=counters),
        "replay_buffer": buffer_state,
        "counters": dict(counters),
        "episode_index": episode_index,
        "episodes": episodes,
        "updates": updates,
        "resume_validation": _validation_state(learner, buffer_state),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    atomic_torch_save(state, output_dir / checkpoint_name)


def _assert_batches_equal(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            _assert_batches_equal(expected_value, actual_value)
        elif not torch.equal(expected_value, actual_value):
            raise ValueError(f"Replay Buffer next sample changed while restoring: {key}")


def restore_training_state(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[
    EXPOLearner,
    EXPOReplayBuffer,
    dict[str, int],
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    validate_checkpoint_format(checkpoint)
    learner, metadata = EXPOLearner.from_checkpoint_state(checkpoint["expo"], device=device)
    buffer = EXPOReplayBuffer.from_checkpoint_state(checkpoint["replay_buffer"])
    counters = dict(checkpoint["counters"])
    if metadata != counters or learner.update_steps.item() != counters["gradient_steps"]:
        raise ValueError("EXPO checkpoint counters do not match")

    validation = checkpoint["resume_validation"]
    if validation is not None:
        pixels = torch.from_numpy(validation["pixels"]).unsqueeze(0).to(device)
        state = torch.from_numpy(validation["state"]).unsqueeze(0).to(device)
        executed_action = torch.from_numpy(validation["executed_action"]).unsqueeze(0).to(device)
        with torch.no_grad():
            residual = learner.act_residual(
                pixels, state, executed_action, deterministic=True
            )[0].cpu()
            embedding = learner.inference_embedding(pixels)
            target_q = learner.target_critic(
                embedding, state, executed_action.reshape(1, 56)
            )[0].cpu()
        if not torch.equal(residual, torch.from_numpy(validation["deterministic_residual"])):
            raise ValueError("Deterministic edit policy output changed while restoring")
        if not torch.equal(target_q, torch.from_numpy(validation["target_q"])):
            raise ValueError("Target Critic output changed while restoring")
        if float(learner.temperature.detach().cpu()) != validation["temperature"]:
            raise ValueError("Temperature changed while restoring")
        buffer_probe = EXPOReplayBuffer.from_checkpoint_state(checkpoint["replay_buffer"])
        _assert_batches_equal(
            validation["next_buffer_batch"], buffer_probe.sample(min(4, len(buffer_probe)))
        )

    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    if torch.cuda.is_available() and checkpoint["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    return (
        learner,
        buffer,
        counters,
        int(checkpoint["episode_index"]),
        list(checkpoint["episodes"]),
        list(checkpoint["updates"]),
        dict(checkpoint["initial_learner_fingerprints"]),
    )


def main() -> None:
    args = parse_args()
    if args.verify_resume and args.resume is None:
        raise ValueError("--verify-resume requires --resume")
    config_path = args.config.resolve()
    config = load_config(config_path)
    resume_path = args.resume.resolve() if args.resume is not None else None
    resume_state = None
    if resume_path is not None:
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_checkpoint_format(resume_state)
        validate_resume_config(resume_state["config"], config)

    seed = int(config["seed"])
    device = torch.device(config["device"])
    checkpoint_path = Path(config["checkpoint"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (checkpoint_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing SmolVLA checkpoint: {checkpoint_path}")

    set_seed(seed)
    environment = config["environment"]
    suite = str(environment["suite"])
    task_id = int(environment["task_id"])
    max_episode_steps = int(environment["max_episode_steps"])
    env_config = make_libero_config(suite, [task_id], max_episode_steps)
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[suite][task_id]

    try:
        stack = make_smolvla_stack(checkpoint_path, env_config, device)
        policy = stack.policy
        preprocessor = stack.preprocessor
        postprocessor = stack.postprocessor
        env_preprocessor = stack.env_preprocessor
        env_postprocessor = stack.env_postprocessor
        current_policy_hash = parameter_hash(policy)

        if resume_state is None:
            learner = EXPOLearner(make_expo_config(config), device=device)
            buffer = EXPOReplayBuffer(
                capacity=int(config["replay_buffer"]["capacity"]), seed=seed
            )
            counters = {"environment_steps": 0, "chunk_transitions": 0, "gradient_steps": 0}
            episode_index = 0
            episodes: list[dict[str, Any]] = []
            updates: list[dict[str, Any]] = []
            initial_fingerprints = learner_fingerprints(learner)
        else:
            if resume_state["smolvla_checkpoint"] != str(checkpoint_path):
                raise ValueError("SmolVLA checkpoint path changed before resume")
            if resume_state["smolvla_parameter_hash"] != current_policy_hash:
                raise ValueError("SmolVLA parameter hash does not match training checkpoint")
            (
                learner,
                buffer,
                counters,
                episode_index,
                episodes,
                updates,
                initial_fingerprints,
            ) = restore_training_state(resume_state, device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training = config["training"]
        batch_size = int(config["expo"]["batch_size"])
        learning_starts = int(training["learning_starts"])
        min_episodes = int(training["min_episodes"])
        min_transitions = int(training["min_chunk_transitions"])
        min_gradient_steps = int(training["min_gradient_steps"])
        save_every_gradient_steps = int(training.get("save_every_gradient_steps", 0))
        if args.verify_resume:
            min_episodes = max(min_episodes, episode_index + 1)
            min_gradient_steps = max(min_gradient_steps, counters["gradient_steps"] + 1)

        def report(status: str) -> dict[str, Any]:
            final_fingerprints = learner_fingerprints(learner)
            changes = {
                name: final_fingerprints[name] != initial_fingerprints[name]
                for name in final_fingerprints
            }
            selected_edited = [
                transition["selected_edited"]
                for episode in episodes
                for transition in episode["transitions"]
            ]
            inference_times = [
                transition["inference_seconds"]
                for episode in episodes
                for transition in episode["transitions"]
            ]
            q_deltas = [
                transition["mean_edited_q_delta"]
                for episode in episodes
                for transition in episode["transitions"]
                if transition.get("mean_edited_q_delta") is not None
            ]
            return {
                "status": status,
                "config_path": str(config_path),
                "resumed_from": str(resume_path) if resume_path is not None else None,
                "resume_verified": bool(args.verify_resume),
                "config": config,
                "smolvla_checkpoint": str(checkpoint_path),
                "smolvla_parameter_hash": current_policy_hash,
                "policy_frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
                "counters": dict(counters),
                "episode_index": episode_index,
                "buffer": buffer.diagnostics(),
                "learner_parameters_changed": changes,
                "edited_selection_rate": (
                    float(np.mean(selected_edited)) if selected_edited else None
                ),
                "mean_candidate_inference_seconds": (
                    float(np.mean(inference_times)) if inference_times else None
                ),
                "mean_edited_q_delta": float(np.mean(q_deltas)) if q_deltas else None,
                "mean_update_seconds": (
                    float(np.mean([update["update_seconds"] for update in updates]))
                    if updates
                    else None
                ),
                "peak_vram_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
                "episodes": episodes,
                "updates": updates,
            }

        while (
            episode_index < min_episodes
            or counters["chunk_transitions"] < min_transitions
            or counters["gradient_steps"] < min_gradient_steps
        ):
            init_state_index = training_init_state_id(episode_index, seed, environment)
            env.set_attr("init_state_id", init_state_index)
            result = rollout_episode(
                env,
                policy,
                learner,
                env_preprocessor,
                preprocessor,
                postprocessor,
                env_postprocessor,
                buffer,
                EXPORolloutConfig(
                    max_episode_steps=max_episode_steps,
                    discount=float(environment["discount"]),
                    bootstrap_on_truncation=bool(environment["bootstrap_on_truncation"]),
                    seed=seed + episode_index,
                    warmup_base_only_until=(
                        learning_starts if training.get("warmup_base_only", False) else 0
                    ),
                    base_exploration_prob=float(
                        training.get("base_exploration_prob", 0.0)
                    ),
                ),
            )
            new_transitions = int(result["chunk_transitions"])
            counters["environment_steps"] += int(result["environment_steps"])
            counters["chunk_transitions"] += new_transitions
            result["episode_index"] = episode_index
            result["init_state_index"] = init_state_index
            result["seed"] = seed + episode_index
            episodes.append(result)

            if len(buffer) >= learning_starts:
                remaining = max(0, min_gradient_steps - counters["gradient_steps"])
                smoke_remaining = max(
                    0,
                    int(training["smoke_gradient_steps"]) - counters["gradient_steps"],
                )
                smoke_updates = min(
                    remaining,
                    smoke_remaining,
                    new_transitions * int(training["smoke_utd_ratio"]),
                )
                regular_updates = min(
                    remaining - smoke_updates,
                    new_transitions * int(training["utd_ratio"]),
                )
                for _ in range(smoke_updates + regular_updates):
                    start = time.perf_counter()
                    metrics = learner.update(
                        buffer.sample(batch_size),
                        update_actor=(
                            counters["gradient_steps"]
                            >= int(training.get("critic_warmup_gradient_steps", 0))
                        ),
                    )
                    update_seconds = time.perf_counter() - start
                    assert_finite(learner, metrics)
                    counters["gradient_steps"] += 1
                    updates.append(
                        {
                            "gradient_step": counters["gradient_steps"],
                            "update_seconds": update_seconds,
                            **metrics,
                        }
                    )
                    if (
                        save_every_gradient_steps > 0
                        and counters["gradient_steps"] % save_every_gradient_steps == 0
                    ):
                        save_training_state(
                            output_dir,
                            learner,
                            buffer,
                            counters,
                            episode_index + 1,
                            episodes,
                            updates,
                            config,
                            current_policy_hash,
                            initial_fingerprints,
                            checkpoint_name=(
                                f"checkpoint_step_{counters['gradient_steps']:06d}.pt"
                            ),
                        )

            assert_finite(learner)
            episode_index += 1
            if episode_index % int(training["save_every_episodes"]) == 0:
                save_training_state(
                    output_dir,
                    learner,
                    buffer,
                    counters,
                    episode_index,
                    episodes,
                    updates,
                    config,
                    current_policy_hash,
                    initial_fingerprints,
                )
                write_json(output_dir / "training.json", report("running"))
            print(
                json.dumps(
                    {
                        "episode": episode_index,
                        "success": result["success"],
                        "episode_steps": result["environment_steps"],
                        **counters,
                    }
                ),
                flush=True,
            )

        final_policy_hash = parameter_hash(policy)
        if final_policy_hash != current_policy_hash:
            raise RuntimeError("SmolVLA parameter hash changed during EXPO training")
        final_fingerprints = learner_fingerprints(learner)
        for name in ("actor", "critic", "image_encoder", "temperature"):
            if final_fingerprints[name] == initial_fingerprints[name]:
                raise RuntimeError(f"EXPO {name} parameters did not update")
        edited_flags = [
            transition["selected_edited"]
            for episode in episodes
            for transition in episode["transitions"]
        ]
        if not any(edited_flags) or all(edited_flags):
            raise RuntimeError("EXPO rollout selected only one candidate class")

        save_training_state(
            output_dir,
            learner,
            buffer,
            counters,
            episode_index,
            episodes,
            updates,
            config,
            current_policy_hash,
            initial_fingerprints,
        )
        final_report = report("passed")
        final_report["final_policy_hash"] = final_policy_hash
        final_report["policy_hash_unchanged"] = True
        write_json(output_dir / "training.json", final_report)
        print(json.dumps({"status": "passed", **counters}, indent=2), flush=True)
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()
