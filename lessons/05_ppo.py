"""第 5 课：PPO 训练、独立评估、保存/加载，以及 MuJoCo 对比回放。"""

import argparse
import csv
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import queue
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pendulum_env import PendulumEnv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "lesson05"
VALIDATION_SEEDS = tuple(range(900, 905))
TEST_SEEDS = tuple(range(1000, 1010))
CONFIG = dict(learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10,
              gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
              vf_coef=0.5, max_grad_norm=0.5, seed=42, device="cpu")
PROBES = np.array([[0.5, 0], [0.2, 0.1], [0, 0], [-0.2, -0.1]], dtype=np.float32)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def pd_action(obs, env):
    # 仅供对照；PPO 不调用这里，也看不到模型的 bias。
    torque = 4 * float(obs[0]) - 3 * float(obs[1]) + float(env.data.qfrc_bias[0])
    return np.array([np.clip(torque / 2, -1, 1)], dtype=np.float32)


def evaluate(model, seeds=TEST_SEEDS):
    """model=None 表示 PD 对照。每个策略使用相同的评估初态种子。"""
    episodes = []
    env = PendulumEnv()
    try:
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            total, trace = 0.0, []
            while True:
                action = pd_action(obs, env) if model is None else model.predict(
                    obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                trace.append([info["error_rad"], info["dq_rad_s"]])
                if terminated or truncated:
                    break
            full = bool(truncated and len(trace) == env.max_episode_steps)
            tail = np.asarray(trace[-50:]) if full else None
            episodes.append({
                "seed": seed, "return": total, "steps": len(trace),
                "terminated": terminated, "truncated": truncated,
                "final_q_rad": info["q_rad"], "final_dq_rad_s": info["dq_rad_s"],
                "tail_max_error_rad": float(np.max(np.abs(tail[:, 0]))) if full else None,
                "tail_rms_speed_rad_s": float(np.sqrt(np.mean(tail[:, 1]**2))) if full else None,
            })
    finally:
        env.close()
    full_episodes = [item for item in episodes if item["tail_max_error_rad"] is not None]
    return {"mean_return": float(np.mean([item["return"] for item in episodes])),
            "std_return": float(np.std([item["return"] for item in episodes])),
            "completed_episodes": len(full_episodes), "episodes_count": len(episodes),
            "failure_count": sum(item["terminated"] for item in episodes),
            "worst_tail_error_rad": max((item["tail_max_error_rad"] for item in full_episodes), default=None),
            "worst_tail_rms_speed_rad_s": max((item["tail_rms_speed_rad_s"] for item in full_episodes), default=None),
            "episodes": episodes}


class LearningTrace(BaseCallback):
    """记录真实采样批次，并用独立验证初态选择 checkpoint。"""
    def __init__(self, run_dir):
        super().__init__()
        self.run_dir = run_dir
        self.best_score = -np.inf
        self.next_evaluation = 10000
        self.validation_rows = []

    def validate(self):
        result = evaluate(self.model, VALIDATION_SEEDS)
        row = {"environment_steps": self.num_timesteps,
               "mean_return": result["mean_return"],
               "completed_episodes": result["completed_episodes"],
               "worst_tail_error_rad": result["worst_tail_error_rad"]}
        self.validation_rows.append(row)
        with (self.run_dir / "validation.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.validation_rows)
        if result["mean_return"] > self.best_score:
            self.best_score = result["mean_return"]
            self.model.save(self.run_dir / "best_model")
        print(json.dumps({"validation": row, "best_return": self.best_score}), flush=True)

    def _on_training_start(self):
        self.validate()

    def _on_step(self):
        if self.num_timesteps >= self.next_evaluation:
            self.validate()
            self.next_evaluation += 10000
        return True

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        # 这些量来自真正的采样和 GAE，不是人为造出的教学数字。
        example = {"environment_steps": self.num_timesteps,
                   "buffer_shape": list(buffer.observations.shape),
                   "observation": buffer.observations[0, 0].tolist(),
                   "sampled_action_before_clipping": buffer.actions[0, 0].tolist(),
                   "reward": float(buffer.rewards[0, 0]),
                   "old_value": float(buffer.values[0, 0]),
                   "old_log_probability": float(buffer.log_probs[0, 0]),
                   "gae_advantage": float(buffer.advantages[0, 0]),
                   "critic_target_return": float(buffer.returns[0, 0])}
        if not (self.run_dir / "batch_example.json").exists():
            save_json(self.run_dir / "batch_example.json", example)
        save_json(self.run_dir / "last_batch_example.json", example)

    def _on_training_end(self):
        self.validate()


def evaluate_saved(run_dir):
    results = {}
    for label, filename in (("initial", "initial.zip"), ("best", "best_model.zip"),
                            ("final", "final.zip")):
        results[label] = evaluate(PPO.load(run_dir / filename, device="cpu"))
    results["pd_reference"] = evaluate(None)
    save_json(run_dir / "evaluation.json", results)
    for label, result in results.items():
        print(json.dumps({"policy": label, **{k: v for k, v in result.items() if k != "episodes"}}), flush=True)
    return results


def train(steps):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir()
    source_dir = run_dir / "sources"
    source_dir.mkdir()
    for name in ("pendulum_env.py", "05_ppo.py"):
        shutil.copy2(Path(__file__).parent / name, source_dir / name)
    shutil.copy2(ROOT / "requirements.txt", source_dir / "requirements.txt")
    print("RUN_DIRECTORY", run_dir, flush=True)

    # 同一进程顺序推进 4 个独立环境；网络对 4 份观察一起计算。
    envs = DummyVecEnv([lambda: Monitor(PendulumEnv()) for _ in range(4)])
    try:
        model = PPO("MlpPolicy", envs, **CONFIG,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-0.5),
                    verbose=0)
        model.set_logger(configure(str(run_dir), ["csv"]))
        model.save(run_dir / "initial")
        initial_parameters = torch.nn.utils.parameters_to_vector(model.policy.parameters()).detach().clone()
        initial_actions = model.predict(PROBES, deterministic=True)[0].copy()
        save_json(run_dir / "config.json", {**CONFIG, "n_envs": 4, "requested_steps": steps,
                                            "pi_layers": [64, 64], "vf_layers": [64, 64],
                                            "validation_seeds": VALIDATION_SEEDS, "test_seeds": TEST_SEEDS,
                                            "torch": torch.__version__, "cuda_build": torch.version.cuda})
        started = time.perf_counter()
        # 真正的学习入口：反复采样 -> GAE -> 多轮小批次梯度更新。
        model.learn(total_timesteps=steps, callback=LearningTrace(run_dir), log_interval=1)
        seconds = time.perf_counter() - started
        model.save(run_dir / "final")

        reloaded = PPO.load(run_dir / "final.zip", device="cpu")
        final_actions = model.predict(PROBES, deterministic=True)[0]
        loaded_actions = reloaded.predict(PROBES, deterministic=True)[0]
        assert np.array_equal(final_actions, loaded_actions), "Save/load changed the policy output"
        final_parameters = torch.nn.utils.parameters_to_vector(model.policy.parameters()).detach()
        parameter_change = float(torch.linalg.vector_norm(final_parameters - initial_parameters))
        assert parameter_change > 0 and np.max(np.abs(final_actions - initial_actions)) > 1e-3

        results = evaluate_saved(run_dir)
        best = results["best"]
        accepted = (best["completed_episodes"] == len(TEST_SEEDS)
                    and best["mean_return"] >= 350
                    and best["worst_tail_error_rad"] is not None
                    and best["worst_tail_error_rad"] <= 0.05
                    and best["worst_tail_rms_speed_rad_s"] <= 0.1)
        checks = {"save_load_same_actions": True, "parameter_change_l2": parameter_change,
                  "initial_probe_actions": initial_actions.tolist(), "final_probe_actions": final_actions.tolist(),
                  "actual_environment_steps": model.num_timesteps, "training_seconds": seconds,
                  "teaching_acceptance_passed": accepted,
                  "acceptance": {"complete_test_episodes": 10, "mean_return_min": 350,
                                 "worst_tail_error_max_rad": 0.05, "worst_tail_rms_speed_max_rad_s": 0.1}}
        save_json(run_dir / "checks.json", checks)
        save_json(run_dir / "source_locations.json", {
            "ppo_update": inspect.getsourcefile(PPO.train),
            "rollout_collection": inspect.getsourcefile(PPO.collect_rollouts),
            "buffer_class": inspect.getsourcefile(type(model.rollout_buffer))})
        (OUTPUT / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), **checks}), flush=True)
        return run_dir
    finally:
        envs.close()


def play(run_dir):
    initial = PPO.load(run_dir / "initial.zip", device="cpu")
    best = PPO.load(run_dir / "best_model.zip", device="cpu")
    keys = queue.SimpleQueue()
    env = PendulumEnv(render_mode="human", model_name="Lesson 05 - PPO trained control")
    env.key_callback = keys.put
    env.viewer_controls = "1: untrained  2: trained best  3: PD  R: restart  Space: pause"
    kind, paused, ended, total = "untrained", True, False, 0.0
    obs, _ = env.reset(seed=1000)
    env.viewer_label = "untrained | PAUSED"
    try:
        env.render()
        print("查看器：1=未训练网络，2=训练后最佳网络，3=PD对照，R=重来，空格=开始/暂停。", flush=True)
        while env.viewer.is_running():
            frame_started = time.perf_counter()
            with env.viewer.lock():
                while not keys.empty():
                    key = keys.get_nowait()
                    if key == 32:
                        paused = not paused
                    elif key in (49, 50, 51, 82, 114):
                        if key in (49, 50, 51):
                            kind = {49: "untrained", 50: "trained", 51: "pd"}[key]
                        obs, _ = env.reset(seed=1000)
                        env.viewer.perturb.active = 0
                        paused, ended, total = False, False, 0.0
                if not paused and not ended:
                    if kind == "pd":
                        action = pd_action(obs, env)
                    else:
                        policy = initial if kind == "untrained" else best
                        action = policy.predict(obs, deterministic=True)[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    ended = terminated or truncated
                    if ended:
                        print(json.dumps({"viewer_policy": kind, "return": total,
                                          "terminated": terminated, "truncated": truncated, **info}), flush=True)
                status = "ENDED - R to restart" if ended else ("PAUSED" if paused else "RUNNING")
                env.viewer_label = f"{kind} | {status} | return={total:.2f}"
            env.render()
            time.sleep(max(0.0, env.control_dt - (time.perf_counter() - frame_started)))
    finally:
        env.close()


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--view", action="store_true", help="对比未训练/已训练/PD 的三维表现")
    modes.add_argument("--evaluate", action="store_true", help="重新加载并评估保存的模型")
    parser.add_argument("--steps", type=int, default=200000, help="训练环境步数（默认 200000）")
    parser.add_argument("--run", type=Path, help="已有 run 文件夹；默认读取 latest_run.txt")
    args = parser.parse_args()
    if args.view or args.evaluate:
        run_dir = args.run or Path((OUTPUT / "latest_run.txt").read_text(encoding="utf-8").strip())
        play(run_dir) if args.view else evaluate_saved(run_dir)
    else:
        if args.steps < 1024 or args.run is not None:
            parser.error("训练 steps 至少为 1024；--run 仅用于查看和评估已有模型")
        train(args.steps)


if __name__ == "__main__":
    main()
