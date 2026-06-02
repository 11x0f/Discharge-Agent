from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List


# Mock database of known dangerous drug interactions
KNOWN_INTERACTIONS = {
    ("warfarin", "aspirin"): "Increased bleeding risk",
    ("warfarin", "ibuprofen"): "Increased bleeding risk",
    ("warfarin", "naproxen"): "Increased bleeding risk",
    ("metformin", "contrast dye"): "Risk of lactic acidosis",
    ("ssri", "tramadol"): "Risk of serotonin syndrome",
    ("lisinopril", "potassium"): "Risk of hyperkalemia",
    ("digoxin", "amiodarone"): "Increased digoxin toxicity",
    ("methotrexate", "nsaid"): "Increased methotrexate toxicity",
    ("clopidogrel", "omeprazole"): "Reduced clopidogrel efficacy",
    ("simvastatin", "amiodarone"): "Increased risk of myopathy",
}


class DrugInteractionInput(BaseModel):
    medications: List[str] = Field(
        ..., description="List of medication names to check for interactions"
    )


class DrugInteractionTool(BaseTool):
    name: str = "Drug Interaction Checker"
    description: str = (
        "Checks a list of medications for known dangerous interactions. "
        "Returns a list of interaction warnings if any are found."
    )
    args_schema: Type[BaseModel] = DrugInteractionInput

    def _run(self, medications: List[str]) -> dict:
        if not medications:
            return {"interactions": [], "message": "No medications provided"}

        # Normalize to lowercase for matching
        normalized = [m.lower().strip() for m in medications]
        found_interactions = []

        # Check every pair
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                drug_a = normalized[i]
                drug_b = normalized[j]

                # Check both orderings
                warning = KNOWN_INTERACTIONS.get(
                    (drug_a, drug_b)
                ) or KNOWN_INTERACTIONS.get((drug_b, drug_a))

                # Also check partial matches (e.g. "warfarin 5mg" matches "warfarin")
                if not warning:
                    for (k1, k2), v in KNOWN_INTERACTIONS.items():
                        if k1 in drug_a and k2 in drug_b:
                            warning = v
                            break
                        if k2 in drug_a and k1 in drug_b:
                            warning = v
                            break

                if warning:
                    found_interactions.append(
                        {
                            "drug_a": medications[i],
                            "drug_b": medications[j],
                            "warning": warning,
                            "action": "FLAG FOR CLINICIAN REVIEW",
                        }
                    )

        if found_interactions:
            return {
                "interactions": found_interactions,
                "message": f"{len(found_interactions)} interaction(s) found - escalate to clinician",
            }
        else:
            return {
                "interactions": [],
                "message": "No known interactions found in mock database",
            }