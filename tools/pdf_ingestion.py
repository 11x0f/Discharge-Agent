import fitz
import base64
import os
import json
from pathlib import Path
from openai import OpenAI
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_cache_path(pdf_path: Path) -> Path:
    cache_dir = pdf_path.parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / (pdf_path.stem + ".json")


def load_from_cache(pdf_path: Path) -> str | None:
    cache_path = get_cache_path(pdf_path)
    if cache_path.exists():
        print(f"[PDF INGESTION] Cache hit: {pdf_path.name} — skipping Vision API")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["text"]
    return None


def save_to_cache(pdf_path: Path, text: str):
    cache_path = get_cache_path(pdf_path)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"text": text}, f, ensure_ascii=False, indent=2)
    print(f"[PDF INGESTION] Cached: {cache_path}")


def extract_pages_as_images(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append({
            "page": page_num + 1,
            "b64": b64
        })
    doc.close()
    return images


def extract_text_from_image(b64_image: str, page_num: int) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a clinical document digitizer. "
                                "Extract ALL text from this medical document page exactly as written. "
                                "Preserve structure: headers, tables, lists, dates, times, values. "
                                "For handwritten text, transcribe as accurately as possible. "
                                "If a word is unclear, write [unclear] but never skip it. "
                                "Do not summarize, interpret, or add anything not on the page. "
                                "Output only the extracted text."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR extracting page {page_num}: {str(e)}]"


class PDFIngestionInput(BaseModel):
    patient_folder: str = Field(
        ..., description="Path to the folder containing patient PDF files"
    )


class PDFIngestionTool(BaseTool):
    name: str = "PDF Ingestion Tool"
    description: str = (
        "Reads all PDF files in a patient folder and extracts their text content "
        "using GPT-4o Vision for accurate handwritten and printed text recognition. "
        "Returns a dictionary mapping each filename to its full extracted text. "
        "Uses local cache to avoid re-processing already extracted PDFs."
    )
    args_schema: Type[BaseModel] = PDFIngestionInput

    def _run(self, patient_folder: str) -> dict:
        folder = Path(patient_folder)

        if not folder.exists():
            return {"error": f"Folder not found: {patient_folder}"}

        pdf_files = list(folder.glob("*.pdf"))

        if not pdf_files:
            return {"error": f"No PDF files found in: {patient_folder}"}

        results = {}

        for pdf_path in pdf_files:
            print(f"\n[PDF INGESTION] Processing: {pdf_path.name}")

            # Try cache first
            cached = load_from_cache(pdf_path)
            if cached:
                results[pdf_path.name] = cached
                continue

            # No cache — call Vision API
            try:
                pages = extract_pages_as_images(str(pdf_path))
                print(f"[PDF INGESTION] Extracted {len(pages)} pages as images")

                full_text = ""
                for page_info in pages:
                    print(f"[PDF INGESTION] Reading page {page_info['page']}/{len(pages)} with GPT-4o Vision...")
                    page_text = extract_text_from_image(
                        page_info["b64"],
                        page_info["page"]
                    )
                    full_text += f"\n\n--- PAGE {page_info['page']} ---\n{page_text}"

                full_text = full_text.strip()

                # Save to cache
                save_to_cache(pdf_path, full_text)

                results[pdf_path.name] = full_text
                print(f"[PDF INGESTION] Done: {pdf_path.name}")

            except Exception as e:
                results[pdf_path.name] = f"[ERROR processing {pdf_path.name}: {str(e)}]"

        return results