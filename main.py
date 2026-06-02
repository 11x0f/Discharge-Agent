import os
import json
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Task, Process

from agents.orchestrator import create_orchestrator_agent
from agents.conflict_detector import create_conflict_detector_agent
from agents.med_reconciler import create_med_reconciler_agent
from agents.summary_writer import create_summary_writer_agent
from tracer import Tracer

load_dotenv()


def run_patient(patient_folder: str):
    patient_id = Path(patient_folder).name
    print(f"\n{'#'*60}")
    print(f"# Processing Patient: {patient_id}")
    print(f"{'#'*60}")

    tracer = Tracer(patient_id=patient_id)
    step_counter = [1]  # mutable counter for closure

    def step_callback(step_output):
        """Called after every agent step by CrewAI."""
        try:
            # Extract info from step output
            if hasattr(step_output, 'tool'):
                action = f"tool_call: {step_output.tool}"
                inputs = {"tool_input": str(step_output.tool_input)[:300]}
                result = str(step_output.tool_output)[:300] if hasattr(step_output, 'tool_output') else "pending"
                next_decision = "evaluate tool result and decide next action"
            else:
                action = "agent_reasoning"
                inputs = {}
                result = str(step_output)[:300]
                next_decision = "continue task"

            tracer.log(
                step=step_counter[0],
                reasoning=f"Agent step {step_counter[0]}",
                action=action,
                inputs=inputs,
                result=result,
                next_decision=next_decision,
            )
            step_counter[0] += 1
        except Exception as e:
            print(f"[TRACER] Step callback error: {e}")

    def task_callback(task_output):
        """Called after every task completes."""
        try:
            tracer.log(
                step=step_counter[0],
                reasoning="Task completed",
                action="task_complete",
                inputs={"task": str(task_output.description)[:200] if hasattr(task_output, 'description') else "unknown"},
                result=str(task_output.raw)[:500] if hasattr(task_output, 'raw') else str(task_output)[:500],
                next_decision="proceed to next task",
            )
            step_counter[0] += 1
        except Exception as e:
            print(f"[TRACER] Task callback error: {e}")

    # --- Agents ---
    orchestrator = create_orchestrator_agent()
    conflict_detector = create_conflict_detector_agent()
    med_reconciler = create_med_reconciler_agent()
    summary_writer = create_summary_writer_agent()

    # --- Tasks ---
    task_ingest = Task(
        description=(
            f"Use the PDF Ingestion Tool to read all PDFs in the folder: '{patient_folder}'. "
            "Return the extracted text from each document clearly labeled by filename. "
            "If a file fails to load, report it explicitly — do not skip silently."
        ),
        expected_output=(
            "A dictionary mapping each PDF filename to its extracted text content. "
            "Any failed or empty files must be clearly noted."
        ),
        agent=orchestrator,
        callback=task_callback,
    )

    task_conflict = Task(
        description=(
            "Review all extracted patient notes provided by the previous task. "
            "Identify and list every contradiction or inconsistency between documents. "
            "Check for: conflicting diagnoses, conflicting medication lists, "
            "inconsistent dates, conflicting lab values, conflicting procedures. "
            "For each conflict found, state: the field, the two conflicting values, "
            "and which documents they came from. "
            "If no conflicts are found, explicitly state that. "
            "Use the Escalation Tool to escalate every conflict found."
        ),
        expected_output=(
            "A structured list of all conflicts found, each with: field name, "
            "conflicting values, and source documents. "
            "If no conflicts, state: 'No conflicts detected.'"
        ),
        agent=conflict_detector,
        context=[task_ingest],
        callback=task_callback,
    )

    task_medications = Task(
        description=(
            "Using the extracted notes, perform medication reconciliation. "
            "1. Extract the full admission medication list. "
            "2. Extract the full discharge medication list. "
            "3. Compare them and identify: added medications, stopped medications, changed doses or frequencies. "
            "4. For each change, check if a reason is documented in the notes. "
            "   If no reason is documented, flag it explicitly for clinician review. "
            "5. Run the Drug Interaction Checker on the full discharge medication list. "
            "6. Use the Escalation Tool for any undocumented changes or interactions found."
        ),
        expected_output=(
            "A structured medication reconciliation report containing: "
            "admission med list, discharge med list, list of changes with documented reasons, "
            "list of changes flagged for missing reasons, and drug interaction findings."
        ),
        agent=med_reconciler,
        context=[task_ingest],
        callback=task_callback,
    )

    task_summary = Task(
        description=(
            "Using all information from the previous tasks — extracted notes, conflict flags, "
            "and medication reconciliation — produce a structured discharge summary draft. "
            "The summary must include these sections:\n"
            "1. Patient Demographics (name, DOB, patient ID)\n"
            "2. Admission & Discharge Dates\n"
            "3. Principal Diagnosis\n"
            "4. Secondary Diagnoses\n"
            "5. Hospital Course\n"
            "6. Procedures\n"
            "7. Discharge Medications (with changes from admission clearly noted)\n"
            "8. Allergies\n"
            "9. Follow-up Instructions\n"
            "10. Pending Results\n"
            "11. Discharge Condition\n"
            "12. Flags for Clinician Review (conflicts, missing data, interactions)\n\n"
            "CRITICAL RULES:\n"
            "- Only use information explicitly found in the source documents.\n"
            "- Mark any field you cannot source as: [MISSING - FLAG FOR CLINICIAN REVIEW]\n"
            "- Mark any pending result as: [PENDING - FLAG FOR CLINICIAN REVIEW]\n"
            "- Mark any conflicted field as: [CONFLICTED - FLAG FOR CLINICIAN REVIEW]\n"
            "- Never invent, infer, or assume any clinical fact.\n"
            "- Always end with: THIS IS A DRAFT. All flagged fields require clinician review."
        ),
        expected_output=(
            "A complete structured discharge summary draft with all 12 sections filled "
            "or explicitly marked as missing/pending/conflicted. "
            "Ends with a clear draft disclaimer."
        ),
        agent=summary_writer,
        context=[task_ingest, task_conflict, task_medications],
        callback=task_callback,
    )

    # --- Crew ---
    crew = Crew(
        agents=[orchestrator, conflict_detector, med_reconciler, summary_writer],
        tasks=[task_ingest, task_conflict, task_medications, task_summary],
        process=Process.sequential,
        verbose=True,
        step_callback=step_callback,
    )

    # --- Run ---
    tracer.log(
        step=0,
        reasoning="Starting agent crew for patient",
        action="crew.kickoff",
        inputs={"patient_folder": patient_folder},
        result="pending",
        next_decision="Run sequential task pipeline",
    )

    try:
        result = crew.kickoff()

        tracer.log(
            step=step_counter[0],
            reasoning="Crew completed all tasks",
            action="save_output",
            inputs={"patient_id": patient_id},
            result="success",
            next_decision="Save summary and trace to output folder",
        )

        # Save summary
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        summary_path = output_dir / f"{patient_id}_discharge_summary.txt"
        with open(summary_path, "w") as f:
            f.write(str(result))
        print(f"\n[OUTPUT] Summary saved to {summary_path}")

    except Exception as e:
        tracer.log(
            step=step_counter[0],
            reasoning="Crew encountered an unrecoverable error",
            action="error_handler",
            inputs={"error": str(e)},
            result="failed",
            next_decision="Log error, save trace, exit gracefully",
        )
        print(f"\n[ERROR] Agent failed for patient {patient_id}: {e}")

    finally:
        tracer.save()


def main():
    data_dir = Path("data")

    if not data_dir.exists():
        print("[ERROR] data/ folder not found.")
        return

    patient_folders = [f for f in data_dir.iterdir() if f.is_dir()]

    if not patient_folders:
        print("[ERROR] No patient folders found inside data/.")
        return

    print(f"\n[MAIN] Found {len(patient_folders)} patient(s): {[f.name for f in patient_folders]}")

    for folder in patient_folders:
        run_patient(str(folder))


if __name__ == "__main__":
    main()