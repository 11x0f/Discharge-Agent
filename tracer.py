# emits step by step trace for each agent action

import json
import datetime
from pathlib import Path


class Tracer:
    def __init__(self, patient_id: str, output_dir: str = "output"):
        self.patient_id = patient_id
        self.steps = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def log(
        self,
        step: int,
        reasoning: str,
        action: str,
        inputs: dict,
        result: str,
        next_decision: str,
    ):
        entry = {
            "step": step,
            "timestamp": datetime.datetime.now().isoformat(),
            "reasoning": reasoning,
            "action": action,
            "inputs": inputs,
            "result": result,
            "next_decision": next_decision,
        }
        self.steps.append(entry)
        self._print(entry)

    def _print(self, entry: dict):
        print(f"\n{'='*60}")
        print(f"STEP {entry['step']} | {entry['timestamp']}")
        print(f"  REASONING    : {entry['reasoning']}")
        print(f"  ACTION       : {entry['action']}")
        print(f"  INPUTS       : {json.dumps(entry['inputs'], indent=4)}")
        print(f"  RESULT       : {entry['result']}")
        print(f"  NEXT DECISION: {entry['next_decision']}")
        print(f"{'='*60}")

    def save(self):
        path = self.output_dir / f"{self.patient_id}_trace.json"
        with open(path, "w") as f:
            json.dump(self.steps, f, indent=2)
        print(f"\n[TRACER] Trace saved to {path}")
        return path