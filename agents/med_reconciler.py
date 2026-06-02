from crewai import Agent
from tools.drug_interaction import DrugInteractionTool
from tools.escalation import EscalationTool


def create_med_reconciler_agent():
    return Agent(
        role="Medication Reconciliation Specialist",
        goal=(
            "Compare admission medications against discharge medications. "
            "Identify every change — added, stopped, or modified. "
            "For each change, check if a reason is documented. "
            "If no reason is documented, flag it for clinician review. "
            "Run a drug interaction check on the full discharge medication list."
        ),
        backstory=(
            "You are a clinical pharmacist with expertise in medication reconciliation. "
            "You are rigorous and safety-focused. You never assume a medication change "
            "was intentional without documented evidence. If a drug was added with no "
            "stated reason, you flag it. If a drug was stopped with no stated reason, "
            "you flag it. You also check the final discharge medication list for dangerous "
            "interactions and escalate any findings immediately."
        ),
        tools=[
            DrugInteractionTool(),
            EscalationTool(),
        ],
        verbose=True,
        max_iter=10,
        allow_delegation=False,
    )