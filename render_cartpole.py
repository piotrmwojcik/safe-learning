import jax
import numpy as np
import mediapy as media

from brax.training.agents.sac import checkpoint
from hydra import compose, initialize

from ss2r import benchmark_suites
from ss2r.algorithms import sbsrl
from ss2r.algorithms.sbsrl import networks as sbsrl_networks


CHECKPOINT = "outputs/2026-10-06/14-26-57/ckpt/000000005000"
OUTPUT = "cartpole_sbsrl.mp4"


def main():
    # Load the same Hydra configuration used for the experiment.
    with initialize(version_base=None, config_path="ss2r/configs"):
        cfg = compose(
            config_name="train_brax",
            overrides=[
                "environment=cartpole_swingup",
                "agent=sbsrl",

                # Must match checkpoint architecture:
                "agent.model_ensemble_size=2",

                # Evaluation/render settings:
                "training.num_eval_envs=1",
                "training.num_eval_episodes=1",
                "training.episode_length=200",
                "training.eval_domain_randomization=false",
                "training.train_domain_randomization=false",
            ],
        )

    # Construct the same evaluation environment.
    train_wrap, eval_wrap = benchmark_suites.get_wrap_env_fn(cfg)
    _, eval_env = benchmark_suites.make(cfg, train_wrap, eval_wrap)

    # Restore checkpoint.
    params = checkpoint.load(CHECKPOINT)

    # Checkpoint layout in SBSRL:
    # params[0] = observation normalizer
    # params[1] = behavior policy parameters
    policy_params = (params[0], params[1])

    # Recreate the SBSRL policy network.
    network_factory = sbsrl.get_network_factory(cfg)

    network = network_factory(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
    )

    make_policy = sbsrl_networks.make_inference_fn(network)
    policy = make_policy(policy_params, deterministic=True)

    # Use the repo's own renderer.
    rng = jax.random.split(
        jax.random.PRNGKey(cfg.training.seed),
        cfg.training.num_eval_envs,
    )

    video = benchmark_suites.render_fns[cfg.environment.task_name](
        eval_env,
        policy,
        cfg.training.episode_length,
        rng,
    )

    video = np.asarray(video)

    fps = int(1.0 / cfg.environment.task_params.ctrl_dt)

    media.write_video(
        OUTPUT,
        video,
        fps=fps,
    )

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()