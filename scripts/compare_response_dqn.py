#!/usr/bin/env python3
import argparse
import json
import importlib.util
import os
import sys
import tempfile
from datetime import datetime

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def load_local_nasim(repo_root_path):
    init_path = os.path.join(repo_root_path, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "nasim",
        init_path,
        submodule_search_locations=[repo_root_path],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["nasim"] = module
    spec.loader.exec_module(module)
    return module


nasim = load_local_nasim(repo_root)
from nasim.agents.dqn_agent import DQNAgent


def _init_action_counts():
    return {
        "service_scan": 0,
        "os_scan": 0,
        "subnet_scan": 0,
        "process_scan": 0,
        "exploit": 0,
        "privilege_escalation": 0,
        "noop": 0,
    }


def _update_action_counts(action, counts):
    if action.is_noop():
        counts["noop"] += 1
    elif action.is_service_scan():
        counts["service_scan"] += 1
    elif action.is_os_scan():
        counts["os_scan"] += 1
    elif action.is_subnet_scan():
        counts["subnet_scan"] += 1
    elif action.is_process_scan():
        counts["process_scan"] += 1
    elif action.is_exploit():
        counts["exploit"] += 1
    elif action.is_privilege_escalation():
        counts["privilege_escalation"] += 1


def train_collect(env, training_steps, seed, dqn_kwargs):
    dqn_agent = DQNAgent(env, seed=seed, **dqn_kwargs)

    episode_records = []
    best_episode = None

    while dqn_agent.steps_done < training_steps:
        remaining = training_steps - dqn_agent.steps_done
        o, _ = env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0.0
        detected_count = 0
        action_counts = _init_action_counts()
        action_sequence = []
        losses = []
        mean_vs = []

        while not done and not env_step_limit_reached and steps < remaining:
            a = dqn_agent.get_egreedy_action(o, dqn_agent.get_epsilon())
            action_obj = env.action_space.get_action(a)
            action_sequence.append(str(action_obj))
            _update_action_counts(action_obj, action_counts)

            next_o, r, done, env_step_limit_reached, info = env.step(a)
            detected_count += int(info.get("detected", False))

            dqn_agent.replay.store(o, a, next_o, r, done)
            dqn_agent.steps_done += 1
            loss, mean_v = dqn_agent.optimize()
            losses.append(loss)
            mean_vs.append(mean_v)

            o = next_o
            episode_return += r
            steps += 1

        goal = env.goal_reached()
        detected_rate = detected_count / steps if steps > 0 else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_mean_v = float(np.mean(mean_vs)) if mean_vs else 0.0

        record = {
            "episode_return": float(episode_return),
            "episode_steps": int(steps),
            "episode_goal": bool(goal),
            "detected_count": int(detected_count),
            "detected_rate": float(detected_rate),
            "avg_loss": avg_loss,
            "avg_mean_v": avg_mean_v,
            "action_counts": dict(action_counts),
        }
        episode_records.append(record)

        if best_episode is None or episode_return > best_episode["episode_return"]:
            best_episode = {
                **record,
                "action_sequence": list(action_sequence),
            }

    eval_return, eval_steps, eval_goal = dqn_agent.run_eval_episode(
        env, render=False, eval_epsilon=dqn_kwargs["final_epsilon"],
        render_mode="ansi"
    )

    results = {
        "episodes": episode_records,
        "best_episode": best_episode,
        "eval_return": float(eval_return),
        "eval_steps": int(eval_steps),
        "eval_goal": bool(eval_goal),
    }
    return results


def save_plots(results_base, results_response, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib not available, skipping plots: {exc}")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(
        [ep["episode_return"] for ep in results_base["episodes"]],
        label="base",
    )
    plt.plot(
        [ep["episode_return"] for ep in results_response["episodes"]],
        label="response",
    )
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("DQN Training: Returns per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "returns_compare.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(
        [ep["episode_steps"] for ep in results_base["episodes"]],
        label="base",
    )
    plt.plot(
        [ep["episode_steps"] for ep in results_response["episodes"]],
        label="response",
    )
    plt.xlabel("Episode")
    plt.ylabel("Episode Steps")
    plt.title("DQN Training: Steps per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "steps_compare.png"))
    plt.close()


def save_csv(results, condition, out_path):
    import csv

    fieldnames = [
        "condition",
        "episode_idx",
        "episode_return",
        "episode_steps",
        "episode_goal",
        "detected_count",
        "detected_rate",
        "avg_loss",
        "avg_mean_v",
        "service_scan",
        "os_scan",
        "subnet_scan",
        "process_scan",
        "exploit",
        "privilege_escalation",
        "noop",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, ep in enumerate(results["episodes"]):
            row = {
                "condition": condition,
                "episode_idx": idx,
                "episode_return": ep["episode_return"],
                "episode_steps": ep["episode_steps"],
                "episode_goal": ep["episode_goal"],
                "detected_count": ep["detected_count"],
                "detected_rate": ep["detected_rate"],
                "avg_loss": ep["avg_loss"],
                "avg_mean_v": ep["avg_mean_v"],
                "service_scan": ep["action_counts"]["service_scan"],
                "os_scan": ep["action_counts"]["os_scan"],
                "subnet_scan": ep["action_counts"]["subnet_scan"],
                "process_scan": ep["action_counts"]["process_scan"],
                "exploit": ep["action_counts"]["exploit"],
                "privilege_escalation": ep["action_counts"][
                    "privilege_escalation"
                ],
                "noop": ep["action_counts"]["noop"],
            }
            writer.writerow(row)


def save_markdown(summary, out_path):
    def _md_best_episode(section_name, best_ep):
        lines = [
            f"### {section_name} Best Episode Path",
            "",
            f"- return: {best_ep['episode_return']}",
            f"- steps: {best_ep['episode_steps']}",
            f"- goal: {best_ep['episode_goal']}",
            f"- detected_count: {best_ep['detected_count']}",
            f"- detected_rate: {best_ep['detected_rate']:.4f}",
            "",
            "Action sequence:",
            "",
        ]
        for idx, action in enumerate(best_ep["action_sequence"], start=1):
            lines.append(f"{idx}. {action}")
        lines.append("")
        return lines

    lines = [
        "# DQN Response Comparison Report",
        "",
        f"- training_steps: {summary['training_steps']}",
        f"- seed: {summary['seed']}",
        "",
        "## Summary",
        "",
        "| condition | eval_return | eval_steps | eval_goal | avg_return_last10 |",
        "|---|---:|---:|---|---:|",
    ]
    for condition in ["base", "response"]:
        data = summary[condition]
        lines.append(
            f"| {condition} | {data['eval_return']} | {data['eval_steps']} | "
            f"{data['eval_goal']} | {data['avg_return_last10']:.4f} |"
        )
    lines.append("")
    lines.extend(_md_best_episode("Base", summary["base"]["best_episode"]))
    lines.extend(_md_best_episode("Response", summary["response"]["best_episode"]))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Compare DQN training with/without response+honeypot."
    )
    parser.add_argument("--training_steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="runs/compare_response")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    dqn_kwargs = dict(
        hidden_sizes=[64, 64],
        lr=0.001,
        batch_size=32,
        target_update_freq=1000,
        replay_size=100000,
        final_epsilon=0.05,
        init_epsilon=1.0,
        exploration_steps=10000,
        gamma=0.99,
        verbose=True,
    )

    base_path = os.path.join(
        repo_root, "scenarios", "benchmark", "small-honeypot.yaml"
    )
    response_path = os.path.join(
        repo_root, "scenarios", "benchmark", "small-honeypot-response.yaml"
    )

    env_base = nasim.load(
        base_path, fully_obs=True, flat_actions=True, flat_obs=True
    )
    env_response = nasim.load(
        response_path, fully_obs=True, flat_actions=True, flat_obs=True
    )

    results_base = train_collect(
        env_base, args.training_steps, args.seed, dqn_kwargs
    )
    results_response = train_collect(
        env_response, args.training_steps, args.seed, dqn_kwargs
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "training_steps": args.training_steps,
        "seed": args.seed,
        "base": {
            "eval_return": results_base["eval_return"],
            "eval_steps": results_base["eval_steps"],
            "eval_goal": results_base["eval_goal"],
            "avg_return_last10": float(
                np.mean(
                    [ep["episode_return"]
                     for ep in results_base["episodes"][-10:]]
                ) if len(results_base["episodes"]) >= 10 else np.mean(
                    [ep["episode_return"] for ep in results_base["episodes"]]
                )
            ),
            "episodes": results_base["episodes"],
            "best_episode": results_base["best_episode"],
        },
        "response": {
            "eval_return": results_response["eval_return"],
            "eval_steps": results_response["eval_steps"],
            "eval_goal": results_response["eval_goal"],
            "avg_return_last10": float(
                np.mean(
                    [ep["episode_return"]
                     for ep in results_response["episodes"][-10:]]
                ) if len(results_response["episodes"]) >= 10 else np.mean(
                    [ep["episode_return"] for ep in results_response["episodes"]]
                )
            ),
            "episodes": results_response["episodes"],
            "best_episode": results_response["best_episode"],
        },
    }

    json_path = os.path.join(out_dir, f"summary_{timestamp}.json")
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=out_dir,
        delete=False,
        prefix=f"summary_{timestamp}_",
        suffix=".json.tmp",
    ) as f:
        json.dump(summary, f, indent=2)
        temp_path = f.name
    os.replace(temp_path, json_path)

    if not args.no_plots:
        save_plots(results_base, results_response, out_dir)

    csv_base = os.path.join(out_dir, f"episodes_base_{timestamp}.csv")
    csv_resp = os.path.join(out_dir, f"episodes_response_{timestamp}.csv")
    save_csv(results_base, "base", csv_base)
    save_csv(results_response, "response", csv_resp)

    md_path = os.path.join(out_dir, f"report_{timestamp}.md")
    save_markdown(summary, md_path)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {json_path}")
    print(f"Saved CSV to: {csv_base}")
    print(f"Saved CSV to: {csv_resp}")
    print(f"Saved report to: {md_path}")
    if not args.no_plots:
        print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
