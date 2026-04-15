from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .loop import AgentLoop
from .types import TaskContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-agent",
        description="Hermes self-evolving autonomous agent MVP CLI",
    )
    parser.add_argument("task", help="Task prompt for the agent")
    parser.add_argument("--session-id", default="cli-session", help="Session id")
    parser.add_argument("--user-id", default="single-user", help="User id")
    parser.add_argument("--rounds", type=int, default=1, help="Number of repeated runs")
    parser.add_argument("--max-steps", type=int, default=8, help="Max steps per run")
    parser.add_argument(
        "--allow-guarded",
        action="store_true",
        help="Allow actions that require guard approval",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full JSON result payload per round",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    agent = AgentLoop()
    results = []
    for idx in range(max(1, args.rounds)):
        context = TaskContext(
            session_id=args.session_id,
            user_id=args.user_id,
            max_steps=args.max_steps,
            allow_guarded_actions=args.allow_guarded,
            metadata={"round": idx + 1},
        )
        result = agent.run(args.task, context)
        results.append(result)
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=True, indent=2))
        else:
            print(
                f"[round {idx + 1}] success={result.success} "
                f"steps={result.metrics['steps_executed']} "
                f"blocked={result.metrics['blocked_count']} "
                f"latency={result.metrics['estimated_latency_ms']}ms "
                f"skills={result.metrics['matched_skills']}"
            )
            print(result.response)
            if result.skill_updates:
                print("skill_updates:", ", ".join(result.skill_updates))

    if args.rounds > 1 and not args.json:
        first = results[0].metrics
        last = results[-1].metrics
        print(
            "replay_summary:",
            f"steps {first['steps_executed']} -> {last['steps_executed']}, "
            f"latency {first['estimated_latency_ms']} -> {last['estimated_latency_ms']}",
        )


if __name__ == "__main__":
    main()
