import re
from difflib import SequenceMatcher


def normalized_edit_distance(draft: str, corrected: str) -> float:
    """
    Computes normalized edit distance between draft and corrected text.
    Returns a value between 0.0 (identical) and 1.0 (completely different).
    Lower = better (less editing needed).
    """
    matcher = SequenceMatcher(None, draft, corrected)
    similarity = matcher.ratio()
    return round(1.0 - similarity, 4)


def section_accuracy(draft: str, corrected: str) -> dict:
    """
    Computes per-section match rate between draft and corrected summary.
    Returns a dict of section -> similarity score.
    """
    sections = [
        "Patient Demographics",
        "Admission & Discharge Dates",
        "Principal Diagnosis",
        "Secondary Diagnoses",
        "Hospital Course",
        "Procedures",
        "Discharge Medications",
        "Allergies",
        "Follow-up Instructions",
        "Pending Results",
        "Discharge Condition",
        "Flags for Clinician Review",
    ]

    def extract_section(text: str, section: str) -> str:
        pattern = rf"{re.escape(section)}.*?(?=\n\d+\.|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(0).strip() if match else ""

    results = {}
    for section in sections:
        draft_section = extract_section(draft, section)
        corrected_section = extract_section(corrected, section)
        if draft_section and corrected_section:
            similarity = SequenceMatcher(
                None, draft_section, corrected_section
            ).ratio()
            results[section] = round(similarity, 4)
        else:
            results[section] = None

    return results


def compute_reward(draft: str, corrected: str) -> dict:
    """
    Computes full reward signal from a (draft, corrected) pair.
    Higher reward = less editing needed = better draft.
    """
    edit_dist = normalized_edit_distance(draft, corrected)
    section_scores = section_accuracy(draft, corrected)

    valid_scores = [s for s in section_scores.values() if s is not None]
    avg_section_accuracy = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

    reward = round(1.0 - edit_dist, 4)

    return {
        "edit_distance": edit_dist,
        "reward": reward,
        "avg_section_accuracy": avg_section_accuracy,
        "section_scores": section_scores,
    }