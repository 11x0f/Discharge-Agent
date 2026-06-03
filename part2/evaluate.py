import json
from pathlib import Path
from part2.reviewer import simulated_review
from part2.reward import compute_reward
from part2.learner import add_to_memory, build_correction_context, get_memory_stats


def run_evaluation(patient_id: str, iterations: int = 5):
    """
    Runs the learning loop for a patient:
    1. Load the agent's original draft
    2. Inject past correction context into reviewer prompt
    3. Simulated doctor reviews it
    4. Compute reward signal
    5. Store (draft, corrected) pair in memory
    6. Repeat - each iteration reviewer has more context
    7. Print improvement curve
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

    for i in range(1, iterations + 1):
        print(f"\n--- Iteration {i}/{iterations} ---")

        # Get correction context from memory
        correction_context = build_correction_context(n_examples=3)

        # Always start from original draft
        # But inject past corrections as context for the reviewer
        if correction_context:
            draft_with_context = (
                f"{correction_context}\n\n"
                f"---\n\n"
                f"Now apply your editing policy to this NEW draft, "
                f"using the above patterns to make fewer corrections needed:\n\n"
                f"{original_draft}"
            )
        else:
            draft_with_context = original_draft

        # Simulated doctor reviews
        print(f"[EVALUATE] Running simulated review (iteration {i})...")
        corrected = simulated_review(draft_with_context)

        # Compute reward against original draft
        reward_signal = compute_reward(original_draft, corrected)
        print(f"[EVALUATE] Edit distance : {reward_signal['edit_distance']}")
        print(f"[EVALUATE] Reward        : {reward_signal['reward']}")
        print(f"[EVALUATE] Avg section   : {reward_signal['avg_section_accuracy']}")

        # Store in memory
        add_to_memory(original_draft, corrected, reward_signal["reward"])

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
    print(f"BEFORE (iteration 1): Edit distance = {first['edit_distance']}")
    print(f"AFTER  (iteration {iterations}): Edit distance = {last['edit_distance']}")
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
    print("\n" + "="*60)
    print("RUNNING PART 2 EVALUATION ON ALL PATIENTS")
    print("="*60)
    
    results_p1 = run_evaluation("patient_1", iterations=5)
    results_p2 = run_evaluation("patient_2", iterations=5)
    
    print("\n" + "="*60)
    print("COMBINED SUMMARY")
    print("="*60)
    print(f"Patient 1 - First iter edit distance: {results_p1[0]['edit_distance']}")
    print(f"Patient 1 - Last iter edit distance : {results_p1[-1]['edit_distance']}")
    print(f"Patient 2 - First iter edit distance: {results_p2[0]['edit_distance']}")
    print(f"Patient 2 - Last iter edit distance : {results_p2[-1]['edit_distance']}")