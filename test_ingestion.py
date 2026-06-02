from tools.pdf_ingestion import PDFIngestionTool

tool = PDFIngestionTool()
result = tool._run(patient_folder="data/patient_2")

for filename, text in result.items():
    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print(f"{'='*60}")
    print(text[:500])  # print first 500 chars of each
    print("...")