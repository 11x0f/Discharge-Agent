from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class DataStatus(str, Enum):
    PRESENT = "present"
    MISSING = "missing - flag for clinician review"
    PENDING = "pending - flag for clinician review"
    CONFLICTED = "conflicted - flag for clinician review"


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    status: str  # "unchanged", "added", "stopped", "changed"
    change_reason: Optional[str] = None
    flag: Optional[str] = None  # set if change has no documented reason


class PendingResult(BaseModel):
    test_name: str
    notes: Optional[str] = None


class ConflictFlag(BaseModel):
    field: str
    values_found: List[str]
    source_notes: List[str]


class DischargeSummary(BaseModel):
    # Demographics
    patient_name: str = DataStatus.MISSING
    date_of_birth: str = DataStatus.MISSING
    patient_id: str = DataStatus.MISSING

    # Dates
    admission_date: str = DataStatus.MISSING
    discharge_date: str = DataStatus.MISSING

    # Diagnoses
    principal_diagnosis: str = DataStatus.MISSING
    secondary_diagnoses: List[str] = Field(default_factory=list)

    # Clinical
    hospital_course: str = DataStatus.MISSING
    procedures: List[str] = Field(default_factory=list)
    discharge_condition: str = DataStatus.MISSING

    # Medications
    admission_medications: List[Medication] = Field(default_factory=list)
    discharge_medications: List[Medication] = Field(default_factory=list)

    # Safety
    allergies: List[str] = Field(default_factory=list)
    drug_interaction_flags: List[str] = Field(default_factory=list)

    # Follow-up
    follow_up_instructions: str = DataStatus.MISSING
    pending_results: List[PendingResult] = Field(default_factory=list)

    # Flags
    conflict_flags: List[ConflictFlag] = Field(default_factory=list)
    clinician_review_flags: List[str] = Field(default_factory=list)

    # Meta
    draft_note: str = "THIS IS A DRAFT. All fields marked missing, pending, or conflicted require clinician review before finalization."