"""第 4 课：检验环境，并用随机策略/已知控制器走一遍 reset -> step 循环。"""

import argparse
import csv
import json
from pathlib import Path
import queue
import time

from gymnasium.utils.env_checker import check_env
import numpy as np

from pendulum_env import PendulumEnv


OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/lesson04"


def choose_action(kind, obs, env, rng):
    if kind == "random":
        return rng.uniform(-1, 1, size=(1,)).astype(np.float32)
    # 这是第三课的已知模型控制器，作为对照，尚未训练任何策略。
    torque = 4.0 * float(obs[0]) - 3.0 * float(obs[1]) + float(env.data.qfrc_bias[0])
    return np.array([np.clip(torque / 2.0, -1, 1)], dtype=np.float32)


def rollout(kind):
    env = PendulumEnv()
    rng = np.random.default_rng(42)
    rows = []
    try:
        obs, info = env.reset(seed=42, options={"q": 0.0, "dq": 0.0})
        episode_return = 0.0
        while True:
            action = choose_action(kind, obs, env, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            # 这里的观测和奖励是动作执行后的结果；action 是刚才执行的动作。
            rows.append([info["simulated_seconds"], float(obs[0]), float(obs[1]),
                         float(action[0]), info["torque_Nm"], reward])
            assert env.observation_space.contains(obs) and env.action_space.contains(action)
            assert env.data.ncon == 0
            if terminated or truncated:
                break
        tail = np.asarray(rows)[-50:]
        result = {"controller": kind, "environment_steps": len(rows),
                  "simulated_seconds": round(info["simulated_seconds"], 9),
                  "episode_return": episode_return,
                  "terminated": terminated, "truncated": truncated,
                  "final_q_rad": info["q_rad"], "final_dq_rad_s": info["dq_rad_s"],
                  "last_50_steps_max_error_rad": float(np.max(np.abs(tail[:, 1])))}
        with (OUTPUT / f"{kind}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["time_after_s", "next_error_rad", "next_speed_rad_s",
                             "applied_action", "applied_torque_Nm", "reward"])
            writer.writerows(rows)
        return result
    finally:
        env.close()


def verify_contract():
    env = PendulumEnv(max_episode_steps=3)
    try:
        check_env(env, skip_render_check=True)  # 官方检查：返回值、随机种子、空间等。
        first, _ = env.reset(seed=123)
        repeated, _ = env.reset(seed=123)
        assert np.array_equal(first, repeated)
        env.reset(seed=42, options={"q": 0.0, "dq": 0.0})
        for index in range(3):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
            assert not terminated and truncated == (index == 2)
        assert abs(info["simulated_seconds"] - 0.06) < 1e-10
        assert env.data.ctrl[0] == 0.0 and abs(info["q_rad"]) > 0  # 零动作仍受重力作用。
        try:
            env.step(np.zeros(1, dtype=np.float32))
        except RuntimeError:
            pass
        else:
            raise AssertionError("An ended episode must require reset")
        # 注入越界初态，仅验证失败结束与时间上限的区别。
        env.reset(options={"q": 1.8, "dq": 0.0})
        _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        assert terminated and not truncated and info["failure_cost"] == 10.0
        env.reset(options={"q": 0.0, "dq": 0.0})
        env.step(np.ones(1, dtype=np.float32))
        assert env.data.ctrl[0] == 2.0  # 动作 1 直接对应 +2 N·m，没有暗加 PD。
    finally:
        env.close()


def show_demo():
    keys = queue.SimpleQueue()
    env = PendulumEnv(render_mode="human")
    env.key_callback = keys.put
    kind, paused, ended = "random", True, False
    rng = np.random.default_rng(42)
    obs, _ = env.reset(seed=42, options={"q": 0.0, "dq": 0.0})
    episode_return = 0.0
    env.viewer_label = "random | PAUSED - Space to start"
    try:
        env.render()
        print("查看器：1=随机动作，2=已知PD+重力补偿，R=重来，空格=开始/暂停。", flush=True)
        while env.viewer.is_running():
            frame_started = time.perf_counter()
            with env.viewer.lock():
                while not keys.empty():
                    key = keys.get_nowait()
                    if key == 32:
                        paused = not paused
                    elif key in (49, 50, 82, 114):
                        if key in (49, 50):
                            kind = "random" if key == 49 else "pd"
                        obs, _ = env.reset(seed=42, options={"q": 0.0, "dq": 0.0})
                        env.viewer.perturb.active = 0
                        rng = np.random.default_rng(42)
                        paused, ended, episode_return = False, False, 0.0
                if not paused and not ended:
                    action = choose_action(kind, obs, env, rng)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    ended = terminated or truncated
                    if ended:
                        print(json.dumps({"viewer_policy": kind,
                                          "episode_return": episode_return,
                                          "terminated": terminated, "truncated": truncated,
                                          **info}), flush=True)
                status = "EPISODE ENDED - R to restart" if ended else (
                    "PAUSED" if paused else "RUNNING")
                env.viewer_label = f"{kind} | {status} | return={episode_return:.2f}"
            env.render()
            time.sleep(max(0.0, env.control_dt - (time.perf_counter() - frame_started)))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="用 MuJoCo 看观察、动作与奖励")
    args = parser.parse_args()
    if args.view:
        show_demo()
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    verify_contract()
    random_result, pd_result = rollout("random"), rollout("pd")
    assert pd_result["truncated"] and not pd_result["terminated"]
    assert pd_result["last_50_steps_max_error_rad"] < 0.002
    assert pd_result["episode_return"] > random_result["episode_return"]
    results = {"checks": "passed", "physics_dt": 0.002, "action_dt": 0.02,
               "cases": [random_result, pd_result]}
    (OUTPUT / "checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
