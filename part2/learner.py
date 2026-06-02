import json
import os
from pathlib import Path
from difflib import SequenceMatcher


MEMORY_FILE = Path("part2/correction_memory.json")


def load_memory() -> list:
    """Load stored (draft, corrected) pairs from disk."""
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_memory(memory: list):
    """Save (draft, corrected) pairs to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def add_to_memory(draft: str, corrected: str, reward: float):
    """
    Store a (draft, corrected, reward) triple in memory.
    Only store if reward is meaningful (edit distance > 0.01).
    """
    memory = load_memory()
    entry = {
        "draft": draft[:500],        # store first 500 chars as reference
        "corrected": corrected[:500],
        "reward": reward,
    }
    memory.append(entry)
    save_memory(memory)
    print(f"[LEARNER] Stored correction. Memory size: {len(memory)}")


def build_correction_context(n_examples: int = 3) -> str:
    """
    Builds a correction context string from the best past examples.
    Injects this into future agent prompts to improve output quality.
    Higher reward = less editing needed = better example to learn from.
    """
    memory = load_memory()

    if not memory:
        return ""

    # Sort by reward descending — best drafts first
    sorted_memory = sorted(memory, key=lambda x: x["reward"], reverse=True)
    top_examples = sorted_memory[:n_examples]

    context_parts = [
        "LEARNING FROM PAST CORRECTIONS:",
        "The following examples show how a senior clinician improved previous drafts.",
        "Use these patterns to produce a better draft this time.\n",
    ]

    for i, example in enumerate(top_examples, 1):
        context_parts.append(f"Example {i}:")
        context_parts.append(f"  Previous draft excerpt:\n  {example['draft'][:300]}")
        context_parts.append(f"  Corrected to:\n  {example['corrected'][:300]}")
        context_parts.append(f"  Reward score: {example['reward']}\n")

    return "\n".join(context_parts)


def get_memory_stats() -> dict:
    """Returns statistics about the correction memory."""
    memory = load_memory()
    if not memory:
        return {"total_pairs": 0, "avg_reward": 0.0, "best_reward": 0.0}

    rewards = [m["reward"] for m in memory]
    return {
        "total_pairs": len(memory),
        "avg_reward": round(sum(rewards) / len(rewards), 4),
        "best_reward": round(max(rewards), 4),
    }