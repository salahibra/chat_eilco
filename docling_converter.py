from docling.document_converter import DocumentConverter
import json
import os

def docx2markdown(file_path:str):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    output_path = "./markdown_docs/" + file_path.split("/")[-1].replace(".docx", ".md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())
def docx2json(file_path:str):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    output_path = "./json_docs/" + file_path.split("/")[-1].replace(".docx", ".json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result.document.export_to_dict(), indent=2, ensure_ascii=False))



docx_folder = "./docx_formation/"
for file_name in os.listdir(docx_folder):
    if file_name.endswith(".docx"):
        file_path = os.path.join(docx_folder, file_name)    
        docx2json(file_path=file_path)

