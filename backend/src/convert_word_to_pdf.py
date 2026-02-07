import os
import subprocess
from pathlib import Path

def convert_word_to_pdf(input_path, output_path):
    subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', input_path, '--outdir', os.path.dirname(output_path)], check=True)

# Define paths relative to this file, not current working directory
script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent
docx_dir = backend_dir / "data" / "docx_files"
pdf_dir = backend_dir / "data" / "pdf_files"

for filename in os.listdir(docx_dir):
    if filename.endswith('.docx'):
        input_file = str(docx_dir / filename)
        output_file = str(pdf_dir / filename.replace('.docx', '.pdf'))
        convert_word_to_pdf(input_file, output_file)
        print(f'Converted {input_file} to {output_file}')