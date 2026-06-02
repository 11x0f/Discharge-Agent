import json
from pathlib import Path
from part2.reviewer import simulated_review
from part2.reward import compute_reward
from part2.learner import add_to_memory, build_correction_context, get_memory_stats


def run_evaluation(patient_id: str, iterations: int = 5):
    """
    Runs the learning loop for a patient:
    1. Load the agent's draft
    2. Simulated doctor reviews it
    3. Compute reward signal
    4. Store (draft, corrected) pair in memory
    5. Repeat, injecting past corrections each time
    6. Print improvement curve
    """
    # Load original draft
    draft_path = Path(f"output/{patient_id}_discharge_summary.txt")
    if not draft_path.exists():
        print(f"[EVALUATE] No draft found at {draft_path}")
        return

    with open(draft_path, "r", encoding="utf-8") as f:
        original_draft = f.read()

    print(f"\n{'='*60}")
    print(f"PART 2 EVALUATION: {patient_id}")
    print(f"Running {iterations} iterations...")
    print(f"{'='*60}\n")

    results = []
    current_draft = original_draft

    for i in range(1, iterations + 1):
        print(f"\n--- Iteration {i}/{iterations} ---")

        # Get correction context from memory
        correction_context = build_correction_context(n_examples=3)

        # If we have context, prepend it to the draft for the reviewer
        if correction_context:
            draft_with_context = f"{correction_context}\n\n---\n\nCURRENT DRAFT:\n{current_draft}"
        else:
            draft_with_context = current_draft

        # Simulated doctor reviews
        print(f"[EVALUATE] Running simulated review...")
        corrected = simulated_review(draft_with_context)

        # Compute reward
        reward_signal = compute_reward(current_draft, corrected)
        print(f"[EVALUATE] Edit distance : {reward_signal['edit_distance']}")
        print(f"[EVALUATE] Reward        : {reward_signal['reward']}")
        print(f"[EVALUATE] Avg section   : {reward_signal['avg_section_accuracy']}")

        # Store in memory
        add_to_memory(current_draft, corrected, reward_signal["reward"])

        # Save corrected version
        corrected_path = Path(f"output/{patient_id}_corrected_iter{i}.txt")
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(corrected)

        results.append({
            "iteration": i,
            "edit_distance": reward_signal["edit_distance"],
            "reward": reward_signal["reward"],
            "avg_section_accuracy": reward_signal["avg_section_accuracy"],
        })

        # Use corrected as next iteration's draft
        current_draft = corrected

    # Print improvement curve
    print(f"\n{'='*60}")
    print("IMPROVEMENT CURVE")
    print(f"{'='*60}")
    print(f"{'Iter':<6} {'Edit Distance':<15} {'Reward':<10} {'Section Acc':<12}")
    print(f"{'-'*45}")
    for r in results:
        print(
            f"{r['iteration']:<6} "
            f"{r['edit_distance']:<15} "
            f"{r['reward']:<10} "
            f"{r['avg_section_accuracy']:<12}"
        )

    # Before vs after
    first = results[0]
    last = results[-1]
    improvement = round(first['edit_distance'] - last['edit_distance'], 4)
    print(f"\n{'='*60}")
    print(f"BEFORE: Edit distance = {first['edit_distance']}")
    print(f"AFTER:  Edit distance = {last['edit_distance']}")
    print(f"IMPROVEMENT: {improvement} reduction in edit distance")
    print(f"{'='*60}")

    # Save results
    results_path = Path(f"output/{patient_id}_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EVALUATE] Results saved to {results_path}")

    # Memory stats
    stats = get_memory_stats()
    print(f"\n[MEMORY] Total pairs stored : {stats['total_pairs']}")
    print(f"[MEMORY] Avg reward         : {stats['avg_reward']}")
    print(f"[MEMORY] Best reward        : {stats['best_reward']}")

    return results


if __name__ == "__main__":
    run_evaluation("patient_2", iterations=5)