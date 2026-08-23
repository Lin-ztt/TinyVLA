#!/usr/bin/env python3
"""Run and resume online DSRL training on LIBERO."""

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

from tinyvla.dsrl import ReplayBuffer, SAC, SACConfig
from tinyvla.dsrl.rollout import RolloutConfig, rollout_episode
from tinyvla.libero import make_libero_config, make_smolvla_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dsrl/train.yaml"),
    )
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as file:
        config = yaml.safe_load(file)
    training = config["training"]
    batch_size = int(config["sac"]["batch_size"])
    learning_starts = int(training["learning_starts"])
    if learning_starts < batch_size:
        raise ValueError("learning_starts must be at least batch_size")
    for name in ("utd_ratio", "min_episodes", "min_gradient_steps", "save_every_episodes"):
        if int(training[name]) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if int(training.get("min_chunk_transitions", 0)) < 0:
        raise ValueError("training.min_chunk_transitions must be non-negative")
    checkpoint_every = int(training.get("checkpoint_every_gradient_steps", 0))
    if checkpoint_every < 0:
        raise ValueError("training.checkpoint_every_gradient_steps must be non-negative")
    return config


def parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)


def assert_finite_metrics(metrics: dict[str, float]) -> None:
    invalid = {name: value for name, value in metrics.items() if not math.isfinite(value)}
    if invalid:
        raise FloatingPointError(f"Non-finite SAC metrics: {invalid}")


def assert_finite_learner(learner: SAC) -> None:
    for name, parameter in learner.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite SAC parameter: {name}")


def validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    for name in ("seed", "checkpoint", "device", "environment", "sac", "replay_buffer"):
        if saved[name] != current[name]:
            raise ValueError(f"Resume config mismatch: {name}")
    for name in ("learning_starts", "utd_ratio", "min_chunk_transitions"):
        if name not in saved["training"] and name not in current["training"]:
            continue
        if saved["training"][name] != current["training"][name]:
            raise ValueError(f"Resume config mismatch: training.{name}")


def restore_rng_state(state: dict[str, Any], noise_rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    noise_rng.bit_generator.state = state["noise"]


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_training_state(
    output_dir: Path,
    learner: SAC,
    buffer: ReplayBuffer,
    counters: dict[str, int],
    episode_index: int,
    episodes: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    config: dict[str, Any],
    policy_hash: str,
    noise_rng: np.random.Generator,
) -> None:
    validation = None
    if len(buffer) > 0:
        pixels = torch.from_numpy(buffer.observations["pixels"][0:1])
        state = torch.from_numpy(buffer.observations["state"][0:1])
        action = learner.act(pixels, state, deterministic=True)[0].cpu().numpy()
        validation = {
            "pixels": buffer.observations["pixels"][0].copy(),
            "state": buffer.observations["state"][0].copy(),
            "deterministic_action": action.copy(),
        }
    state = {
        "format_version": 1,
        "config": config,
        "smolvla_checkpoint": str(Path(config["checkpoint"]).resolve()),
        "smolvla_parameter_hash": policy_hash,
        "sac": learner.checkpoint_state(metadata=counters),
        "replay_buffer": buffer.checkpoint_state(),
        "counters": dict(counters),
        "episode_index": episode_index,
        "episodes": episodes,
        "updates": updates,
        "resume_validation": validation,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "noise": noise_rng.bit_generator.state,
        },
    }
    atomic_torch_save(state, output_dir / "checkpoint_latest.pt")

    sac_path = output_dir / "sac_latest.pt"
    sac_temporary = sac_path.with_suffix(sac_path.suffix + ".tmp")
    learner.save_checkpoint(sac_temporary, metadata=counters)
    sac_temporary.replace(sac_path)
    buffer_path = output_dir / "replay_buffer_latest.npz"
    buffer_temporary = buffer_path.with_suffix(buffer_path.suffix + ".tmp")
    buffer.save(buffer_temporary)
    buffer_temporary.replace(buffer_path)


def save_sac_milestone(
    output_dir: Path,
    learner: SAC,
    counters: dict[str, int],
    environment_config: dict[str, Any],
) -> None:
    gradient_steps = counters["gradient_steps"]
    path = output_dir / f"sac_step_{gradient_steps:06d}.pt"
    temporary = path.with_suffix(path.suffix + ".tmp")
    learner.save_checkpoint(
        temporary,
        metadata={
            "counters": dict(counters),
            "environment": dict(environment_config),
        },
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    resume_path = args.resume.resolve() if args.resume is not None else None
    resume_state = None
    if resume_path is not None:
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_state.get("format_version") != 1:
            raise ValueError("Unsupported training checkpoint format")
        validate_resume_config(resume_state["config"], config)
    environment_config = config["environment"]
    sac_config = config["sac"]
    training_config = config["training"]
    seed = int(config["seed"])
    device = torch.device(config["device"])
    checkpoint = Path(config["checkpoint"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing SmolVLA checkpoint: {checkpoint}")

    set_seed(seed)
    suite = str(environment_config["suite"])
    task_id = int(environment_config["task_id"])
    max_episode_steps = int(environment_config["max_episode_steps"])
    execute_horizon = int(environment_config["execute_horizon"])
    discount = float(environment_config["discount"])
    sac_image_keys = tuple(
        environment_config.get("sac_image_keys", ["observation.images.image"])
    )
    image_channels = 3 * len(sac_image_keys)
    env_config = make_libero_config(suite, [task_id], max_episode_steps)
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[suite][task_id]

    try:
        stack = make_smolvla_stack(checkpoint, env_config, device)
        policy = stack.policy
        preprocessor = stack.preprocessor
        postprocessor = stack.postprocessor
        env_preprocessor = stack.env_preprocessor
        env_postprocessor = stack.env_postprocessor
        current_policy_hash = parameter_hash(policy)
        if any(parameter.requires_grad for parameter in policy.parameters()):
            raise RuntimeError("SmolVLA must be frozen before online training")
        rng = np.random.default_rng(seed)
        if resume_state is None:
            learner = SAC(
                SACConfig(
                    image_shape=(image_channels, 64, 64),
                    actor_lr=float(sac_config["actor_lr"]),
                    critic_lr=float(sac_config["critic_lr"]),
                    temperature_lr=float(sac_config["temperature_lr"]),
                    discount=discount,
                    tau=float(sac_config["tau"]),
                    initial_temperature=float(sac_config["initial_temperature"]),
                    target_entropy=float(sac_config["target_entropy"]),
                    num_qs=int(sac_config["num_qs"]),
                    random_crop_padding=int(sac_config["random_crop_padding"]),
                    color_jitter=bool(sac_config.get("color_jitter", False)),
                ),
                device=device,
            )
            buffer = ReplayBuffer(
                capacity=int(config["replay_buffer"]["capacity"]),
                seed=seed,
                image_channels=image_channels,
            )
            initial_policy_hash = current_policy_hash
            counters = {
                "environment_steps": 0,
                "chunk_transitions": 0,
                "gradient_steps": 0,
            }
            episodes: list[dict[str, Any]] = []
            updates: list[dict[str, Any]] = []
            episode_index = 0
        else:
            if resume_state["smolvla_checkpoint"] != str(checkpoint):
                raise ValueError("SmolVLA checkpoint path changed before resume")
            initial_policy_hash = resume_state["smolvla_parameter_hash"]
            if current_policy_hash != initial_policy_hash:
                raise ValueError("SmolVLA parameter hash does not match training checkpoint")
            learner, metadata = SAC.from_checkpoint_state(resume_state["sac"], device=device)
            buffer = ReplayBuffer.from_checkpoint_state(resume_state["replay_buffer"])
            counters = dict(resume_state["counters"])
            if metadata != counters:
                raise ValueError("SAC metadata and training counters do not match")
            episodes = list(resume_state["episodes"])
            updates = list(resume_state["updates"])
            episode_index = int(resume_state["episode_index"])
            if learner.update_steps.item() != counters["gradient_steps"]:
                raise ValueError("SAC update steps and training counters do not match")
            if episode_index != len(episodes):
                raise ValueError("Episode index and saved episode history do not match")
            validation = resume_state["resume_validation"]
            expected_action = torch.from_numpy(validation["deterministic_action"])
            restored_action = learner.act(
                torch.from_numpy(validation["pixels"]).unsqueeze(0),
                torch.from_numpy(validation["state"]).unsqueeze(0),
                deterministic=True,
            )[0].cpu()
            if not torch.equal(restored_action, expected_action):
                raise ValueError("Deterministic Actor output changed while restoring checkpoint")
            restore_rng_state(resume_state["rng"], rng)
        env.set_attr("init_state_id", episode_index)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        batch_size = int(sac_config["batch_size"])
        learning_starts = int(training_config["learning_starts"])
        utd_ratio = int(training_config["utd_ratio"])

        def random_noise(_: dict[str, np.ndarray]) -> np.ndarray:
            return rng.standard_normal(32).astype(np.float32)

        def actor_noise(observation: dict[str, np.ndarray], deterministic: bool) -> np.ndarray:
            pixels = torch.from_numpy(observation["pixels"]).unsqueeze(0)
            state = torch.from_numpy(observation["state"]).unsqueeze(0)
            return (
                learner.act(pixels, state, deterministic=deterministic)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

        def report(status: str) -> dict[str, Any]:
            peak_vram = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            return {
                "status": status,
                "config_path": str(config_path),
                "resumed_from": str(resume_path) if resume_path is not None else None,
                "config": config,
                "checkpoint": str(checkpoint),
                "policy_frozen": not any(
                    parameter.requires_grad for parameter in policy.parameters()
                ),
                "initial_policy_hash": initial_policy_hash,
                "episode_index": episode_index,
                "counters": dict(counters),
                "buffer": buffer.diagnostics(),
                "episodes": episodes,
                "updates": updates,
                "peak_vram_bytes": peak_vram,
            }

        min_episodes = int(training_config["min_episodes"])
        min_gradient_steps = int(training_config["min_gradient_steps"])
        min_chunk_transitions = int(training_config.get("min_chunk_transitions", 0))
        checkpoint_every_gradient_steps = int(
            training_config.get("checkpoint_every_gradient_steps", 0)
        )
        while (
            episode_index < min_episodes
            or counters["gradient_steps"] < min_gradient_steps
            or counters["chunk_transitions"] < min_chunk_transitions
        ):
            episode_seed = seed + episode_index
            init_state_index = episode_index
            use_actor = counters["gradient_steps"] > 0
            result = rollout_episode(
                env,
                policy,
                env_preprocessor,
                preprocessor,
                postprocessor,
                env_postprocessor,
                buffer,
                RolloutConfig(
                    execute_horizon=execute_horizon,
                    max_episode_steps=max_episode_steps,
                    discount=discount,
                    bootstrap_on_truncation=bool(
                        environment_config["bootstrap_on_truncation"]
                    ),
                    seed=episode_seed,
                    sac_image_keys=sac_image_keys,
                ),
                noise_fn=(
                    (lambda observation: actor_noise(observation, deterministic=False))
                    if use_actor
                    else random_noise
                ),
                noise_source="actor_stochastic" if use_actor else "structured_gaussian",
            )
            new_transitions = int(result["chunk_transitions"])
            counters["environment_steps"] += int(result["environment_steps"])
            counters["chunk_transitions"] += new_transitions
            result["episode_index"] = episode_index
            result["init_state_index"] = init_state_index
            result["seed"] = episode_seed
            result["noise_source"] = "actor_stochastic" if use_actor else "structured_gaussian"
            episodes.append(result)

            if len(buffer) >= learning_starts:
                remaining_updates = min_gradient_steps - counters["gradient_steps"]
                for _ in range(
                    min(new_transitions * utd_ratio, max(0, remaining_updates))
                ):
                    update_start = time.perf_counter()
                    metrics = learner.update(buffer.sample(batch_size))
                    update_seconds = time.perf_counter() - update_start
                    assert_finite_metrics(metrics)
                    counters["gradient_steps"] += 1
                    updates.append(
                        {
                            "gradient_step": counters["gradient_steps"],
                            "update_seconds": update_seconds,
                            **metrics,
                        }
                    )
                    if (
                        checkpoint_every_gradient_steps > 0
                        and counters["gradient_steps"] % checkpoint_every_gradient_steps == 0
                    ):
                        save_sac_milestone(
                            output_dir,
                            learner,
                            counters,
                            environment_config,
                        )
            assert_finite_learner(learner)
            episode_index += 1
            if episode_index % int(training_config["save_every_episodes"]) == 0:
                save_training_state(
                    output_dir,
                    learner,
                    buffer,
                    counters,
                    episode_index,
                    episodes,
                    updates,
                    config,
                    initial_policy_hash,
                    rng,
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

        evaluation = None
        if bool(training_config["eval_at_end"]):
            evaluation_buffer = ReplayBuffer(
                capacity=(max_episode_steps + execute_horizon - 1) // execute_horizon,
                seed=seed,
                image_channels=image_channels,
            )
            evaluation = rollout_episode(
                env,
                policy,
                env_preprocessor,
                preprocessor,
                postprocessor,
                env_postprocessor,
                evaluation_buffer,
                RolloutConfig(
                    execute_horizon=execute_horizon,
                    max_episode_steps=max_episode_steps,
                    discount=discount,
                    bootstrap_on_truncation=bool(
                        environment_config["bootstrap_on_truncation"]
                    ),
                    seed=seed + 100_000,
                    sac_image_keys=sac_image_keys,
                ),
                noise_fn=lambda observation: actor_noise(observation, deterministic=True),
                noise_source="actor_deterministic",
            )

        final_policy_hash = parameter_hash(policy)
        if final_policy_hash != initial_policy_hash:
            raise RuntimeError("SmolVLA parameter hash changed during online training")
        save_training_state(
            output_dir,
            learner,
            buffer,
            counters,
            episode_index,
            episodes,
            updates,
            config,
            initial_policy_hash,
            rng,
        )
        final_report = report("passed")
        final_report["final_policy_hash"] = final_policy_hash
        final_report["policy_hash_unchanged"] = True
        final_report["evaluation"] = evaluation
        write_json(output_dir / "training.json", final_report)
        print(json.dumps({"status": "passed", **counters}, indent=2), flush=True)
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()
