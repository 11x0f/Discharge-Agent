from crewai import Agent
from tools.pdf_ingestion import PDFIngestionTool
from tools.drug_interaction import DrugInteractionTool
from tools.escalation import EscalationTool


def create_orchestrator_agent():
    return Agent(
        role="Clinical Discharge Summary Orchestrator",
        goal=(
            "Coordinate the extraction, analysis, and summarization of patient source notes "
            "into a structured discharge summary draft. Plan each step carefully, decide which "
            "tools to use, and ensure no clinical fact is invented or assumed."
        ),
        backstory=(
            "You are a senior clinical informatics specialist with deep experience in hospital "
            "discharge workflows. You are meticulous, cautious, and never guess. If information "
            "is missing, you say so explicitly. If notes conflict, you flag the conflict. "
            "You coordinate a team of specialist agents and tools to produce safe, accurate drafts."
        ),
        tools=[
            PDFIngestionTool(),
            DrugInteractionTool(),
            EscalationTool(),
        ],
        verbose=True,
        max_iter=20,  # hard step cap — agent cannot run forever
        allow_delegation=True,
    )