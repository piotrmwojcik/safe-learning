"""Diagnostic closed-loop Cartpole rollouts, independent of training rollouts."""

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw


def rollout_models(model_apply, policy, normalizer, params, initial_obs, key, steps):
    """Keep each model paired with its own state; return raw observations."""
    count = jax.tree_util.tree_leaves(params)[0].shape[0]
    initial = jnp.broadcast_to(initial_obs, (count, *initial_obs.shape))

    def step(carry, unused):
        obs, rng = carry
        rng, action_key = jax.random.split(rng)
        actions, _ = policy(obs, action_key)
        # Inputs are normalized inside model_apply. Its output is supervised
        # against physical observations via normalize(output) vs normalize(target)
        # in compute_model_loss. Do not denormalize or normalize again here.
        next_obs, reward, cost = jax.vmap(model_apply, in_axes=(None, 0, 0, 0))(
            normalizer, params, obs, actions
        )
        return (next_obs, rng), (next_obs, actions, reward, cost)

    _, (obs, actions, rewards, costs) = jax.lax.scan(
        step, (initial, key), None, length=steps
    )
    return jnp.concatenate([initial[None], obs]), actions, rewards, costs


def cartpole_frames(observations, labels, dt, stride, position_bound, step):
    """Draw [time, world, (x, cos(theta), sin(theta), velocities...)] arrays.

    Pose drawing uses atan2 for orientation only. Raw observations are never
    projected onto a physical state or fed back from the drawing.
    """
    width, height = 280, 300
    count = len(labels)
    for t in range(0, len(observations), stride):
        image = Image.new("RGB", (width * count, height), "white")
        draw = ImageDraw.Draw(image)
        for i, label in enumerate(labels):
            left = width * i
            center, ground, scale = left + width // 2, 185, 65
            draw.text((left + 8, 8), label, fill="black")
            draw.text(
                (left + 8, 26), f"Training step {step} | t={t * dt:.2f}s", fill="black"
            )
            draw.line((left, 0, left, height), fill="gray")
            draw.line(
                (left + 8, ground + 12, left + width - 8, ground + 12), fill="gray"
            )
            for bound in (-position_bound, position_bound):
                x_bound = center + scale * bound
                draw.line((x_bound, 65, x_bound, ground + 20), fill="lightcoral")
            obs = observations[t, i]
            if not np.isfinite(obs).all():
                draw.text((left + 12, 100), "NON-FINITE PREDICTION", fill="red")
                continue
            x, cosine, sine = obs[:3]
            norm = np.hypot(cosine, sine)
            draw.text(
                (left + 8, 235), f"x={x:.3g} | angle norm={norm:.3g}", fill="black"
            )
            cart_x = center + scale * float(x)
            if abs(cart_x - center) > width / 2 - 25:
                draw.text((left + 12, 100), "CART OUTSIDE VIEW", fill="red")
                continue
            angle = np.arctan2(sine, cosine)
            color = "firebrick" if abs(x) >= position_bound else "steelblue"
            draw.rectangle(
                (cart_x - 16, ground - 8, cart_x + 16, ground + 8), fill=color
            )
            if norm > 1e-8:
                tip = (cart_x + 75 * np.sin(angle), ground - 75 * np.cos(angle))
                draw.line((cart_x, ground, *tip), fill="black", width=5)
            else:
                draw.text((left + 12, 80), "UNDEFINED POLE ANGLE", fill="red")
            draw.text((left + 8, 270), "Imagined" if i else "Simulator", fill="gray")
        yield np.asarray(image)


class CartpoleModelRecorder:
    def __init__(self, steps=200, dt=0.01, seed=0, position_bound=0.25):
        if steps < 1 or dt <= 0:
            raise ValueError("Video steps and timestep must be positive")
        self.steps, self.dt, self.seed = steps, dt, seed
        self.position_bound = position_bound
        self._run = None

    def __call__(self, networks, training_state, env, make_policy, step, num_envs):
        # Build once so evaluation intervals reuse the same compiled functions.
        if self._run is None:

            def run(normalizer, policy_params, model_params):
                key = jax.random.PRNGKey(self.seed)
                reset_key, model_key, simulator_key = jax.random.split(key, 3)
                state = env.reset(jax.random.split(reset_key, num_envs))
                policy = make_policy((normalizer, policy_params), deterministic=True)
                imagined = rollout_models(
                    networks.model_network.apply,
                    policy,
                    normalizer,
                    model_params,
                    state.obs[0],
                    model_key,
                    self.steps,
                )

                def simulator_step(carry, unused):
                    state, rng = carry
                    rng, action_key = jax.random.split(rng)
                    actions, _ = policy(state.obs, action_key)
                    next_state = env.step(state, actions)
                    return (next_state, rng), (next_state.obs[0], next_state.done[0])

                _, (sim_obs, done) = jax.lax.scan(
                    simulator_step, (state, simulator_key), None, length=self.steps
                )
                return imagined, jnp.concatenate([state.obs[:1], sim_obs]), done

            self._run = jax.jit(run)

        imagined, simulator, done = jax.device_get(
            self._run(
                training_state.normalizer_params,
                training_state.behavior_policy_params,
                training_state.model_params,
            )
        )
        obs, actions, rewards, costs = imagined
        # Stop before an auto-reset observation: it would not be a continuation
        # of the same initial trajectory used by the learned worlds.
        endings = np.flatnonzero(done)
        length = int(endings[0]) + 1 if len(endings) else len(simulator)
        obs, simulator = obs[:length], simulator[:length]
        output = Path("model_rollouts")
        output.mkdir(parents=True, exist_ok=True)
        stem = output / f"step_{step:012d}"
        np.savez_compressed(
            stem.with_suffix(".npz"),
            observations=obs,
            simulator=simulator,
            actions=actions[: length - 1],
            rewards=rewards[: length - 1],
            costs=costs[: length - 1],
            dt=self.dt,
        )
        combined = np.concatenate([simulator[:, None], obs], axis=1)
        labels = ["Simulator"] + [f"World model {i}" for i in range(obs.shape[1])]
        stride = max(1, round(1 / (25 * self.dt)))
        media.write_video(
            stem.with_suffix(".mp4"),
            cartpole_frames(
                combined, labels, self.dt, stride, self.position_bound, step
            ),
            fps=1 / (stride * self.dt),
        )
        logging.info(
            "Saved model rollout video: %s", stem.with_suffix(".mp4").resolve()
        )
