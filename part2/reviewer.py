import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REVIEWER_SYSTEM_PROMPT = """
You are an experienced senior clinician reviewing an AI-generated discharge summary draft.
You apply a consistent editing policy every time:

1. DEMOGRAPHICS: If name/DOB/ID are missing, add placeholder text: "To be completed by admitting physician"
2. DIAGNOSES: Always list principal diagnosis first, secondary diagnoses as numbered list. 
   If conflicted, keep the conflict flag but add: "Most likely: [pick the clinically most probable based on available evidence]"
3. HOSPITAL COURSE: Must be a single coherent paragraph in past tense. 
   Expand bullet points into prose if needed.
4. MEDICATIONS: Every medication must have: name, dose, route, frequency, duration. 
   If dose is missing, add "[dose to be verified]"
5. ALLERGIES: If "not known", change to "NKDA (No Known Drug Allergies)"
6. FOLLOW-UP: Must include specific date, specific tests, and who to follow up with.
7. FLAGS: Consolidate all flags into a numbered list at the end.
8. ALWAYS end with: "REVIEWED DRAFT - Requires final clinician sign-off before release."

Apply these edits consistently. Do not invent clinical facts.
Only improve structure, completeness markers, and formatting.
"""


def simulated_review(draft: str) -> str:
    """
    Applies consistent editing policy to a discharge summary draft.
    Returns the corrected version.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": REVIEWER_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Please review and correct this discharge summary draft:\n\n{draft}"
                }
            ],
            max_tokens=4000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[REVIEWER] Error: {e}")
        return draft