import functools
from pathlib import Path

import jax
import jax.nn as jnn
import numpy as np
import mediapy as media

from brax.training.agents.sac import checkpoint
from hydra import compose, initialize

from ss2r import benchmark_suites
from ss2r.algorithms.sbsrl import networks as sbsrl_networks


RUN_DIR = Path("outputs/2026-10-08/13-41-05")
OUTPUT = "cartpole_sbsrl.mp4"


def get_latest_checkpoint(run_dir: Path) -> str:
    ckpt_dir = run_dir / "ckpt"

    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()]

    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    latest = max(checkpoints, key=lambda p: int(p.name))
    return str(latest.resolve())


def rollout_cost(
    eval_env,
    policy,
    key,
    episode_length: int,
) -> float:
    """Run one deterministic rollout and return total environment cost."""

    state = eval_env.reset(key)
    total_cost = 0.0

    for _ in range(episode_length):
        key, action_key = jax.random.split(key)

        action, _ = policy(
            state.obs,
            action_key,
        )

        state = eval_env.step(state, action)

        # Cost is stored in state.info by SafeCartpoleSwingup.
        cost = state.info["cost"]

        total_cost += float(np.asarray(cost).sum())

    return total_cost


def main():
    checkpoint_path = get_latest_checkpoint(RUN_DIR)
    print(f"Loading checkpoint: {checkpoint_path}")

    # Reconstruct training configuration.
    with initialize(
        version_base=None,
        config_path="ss2r/configs",
    ):
        cfg = compose(
            config_name="train_brax",
            overrides=[
                "environment=cartpole_swingup",
                "agent=sbsrl",
                # Must match training run.
                "agent.model_ensemble_size=5",
                # Render/evaluation settings.
                "training.num_eval_envs=1",
                "training.num_eval_episodes=1",
                "training.episode_length=500",
                "training.eval_domain_randomization=false",
                "training.train_domain_randomization=false",
            ],
        )

    # Build evaluation environment.
    train_wrap, eval_wrap = benchmark_suites.get_wrap_env_fn(cfg)

    envs = benchmark_suites.make(
        cfg,
        train_wrap,
        eval_wrap,
    )

    assert envs is not None
    _, eval_env = envs

    # Restore checkpoint.
    params = checkpoint.load(checkpoint_path)

    # SBSRL checkpoint layout:
    # params[0] = observation normalizer
    # params[1] = behavior policy parameters
    policy_params = (
        params[0],
        params[1],
    )

    # Recreate policy network.
    value_obs_key = "privileged_state" if cfg.training.value_privileged else "state"

    policy_obs_key = "privileged_state" if cfg.training.policy_privileged else "state"

    network_factory = functools.partial(
        sbsrl_networks.make_sbsrl_networks,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        model_hidden_layer_sizes=cfg.agent.model_hidden_layer_sizes,
        activation=getattr(jnn, cfg.agent.activation),
        value_obs_key=value_obs_key,
        policy_obs_key=policy_obs_key,
    )

    network = network_factory(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        n_critics=cfg.agent.n_critics,
        safe=cfg.training.safe,
        uncertainty_constraint=cfg.agent.uncertainty_constraint,
        use_bro=cfg.agent.use_bro,
        ensemble_size=cfg.agent.model_ensemble_size,
        embedding_dim=cfg.agent.embedding_dim,
    )

    make_policy = sbsrl_networks.make_inference_fn(network)

    policy = make_policy(
        policy_params,
        deterministic=True,
    )

    # Generate 5 rollout keys because the renderer currently returns 5 videos.
    num_rollouts = 5

    rollout_keys = jax.random.split(
        jax.random.PRNGKey(cfg.training.seed),
        num_rollouts,
    )

    # Render.
    video = benchmark_suites.render_fns[cfg.environment.task_name](
        eval_env,
        policy,
        cfg.training.episode_length,
        rollout_keys,
    )

    video = np.asarray(video)

    print("RAW VIDEO SHAPE:", video.shape)

    # Compute costs using the same initial rollout keys.
    costs = []

    for i, key in enumerate(rollout_keys):
        cost = rollout_cost(
            eval_env,
            policy,
            key,
            cfg.training.episode_length,
        )

        costs.append(cost)

        print(f"Rollout {i}: " f"total cost = {cost:.0f}")

    # Save returned videos.
    if video.ndim == 5:
        # Expected:
        # (N, T, C, H, W)

        for i, frames in enumerate(video):
            print(f"Processing rollout {i}, " f"shape before transpose: {frames.shape}")

            # (T, C, H, W) -> (T, H, W, C)
            if frames.ndim == 4 and frames.shape[1] in (1, 3, 4):
                frames = np.moveaxis(
                    frames,
                    1,
                    -1,
                )

            print(f"Rollout {i}, " f"shape after transpose: {frames.shape}")

            cost = costs[i]

            output_i = f"cartpole_sbsrl_{i}" f"_cost_{int(cost)}.mp4"

            media.write_video(
                output_i,
                frames,
                fps=30,
            )

            print(f"Saved {output_i} " f"(cost={cost:.0f})")

    elif video.ndim == 4:
        # Single video case.
        if video.shape[1] in (1, 3, 4):
            video = np.moveaxis(
                video,
                1,
                -1,
            )

        media.write_video(
            OUTPUT,
            video,
            fps=30,
        )

        print(f"Saved {OUTPUT}, " f"cost={costs[0]:.0f}")

    else:
        raise ValueError(f"Unexpected video shape: {video.shape}")


if __name__ == "__main__":
    main()
