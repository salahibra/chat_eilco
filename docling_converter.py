from docling.document_converter import DocumentConverter

def docx2markdown(file_path:str):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    # save result to a markdown file in mardown_docs folder
    output_path = "./markdown_docs/" + file_path.split("/")[-1].replace(".docx", ".md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())


file_path = "/home/salah/EILCO/PIC/chateilco/docx_formation/Guide-des-Etudes_EIL_Cycle-Preparatoire_2024-2025.docx"
docx2markdown(file_path=file_path)