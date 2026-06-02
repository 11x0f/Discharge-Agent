from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List


class EscalationInput(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    flags: List[str] = Field(..., description="List of issues to escalate to clinician")


class EscalationTool(BaseTool):
    name: str = "Escalation Tool"
    description: str = (
        "Escalates one or more clinical flags to the clinician review queue. "
        "Call this whenever there is missing data, a conflict between notes, "
        "an undocumented medication change, or a drug interaction warning. "
        "Never suppress or skip escalation — always call this tool when in doubt."
    )
    args_schema: Type[BaseModel] = EscalationInput

    def _run(self, patient_id: str, flags: List[str]) -> dict:
        if not flags:
            return {"status": "nothing to escalate"}

        escalated = []
        for flag in flags:
            entry = {
                "patient_id": patient_id,
                "flag": flag,
                "action": "REQUIRES CLINICIAN REVIEW BEFORE FINALIZATION",
            }
            escalated.append(entry)
            print(f"\n[ESCALATION] Patient {patient_id}: {flag}")

        return {
            "status": "escalated",
            "count": len(escalated),
            "items": escalated,
        }