import os
import subprocess

def convert_word_to_pdf(input_path, output_path):
    subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', input_path, '--outdir', os.path.dirname(output_path)], check=True)
    
for filename in os.listdir('./docx_files'):
    if filename.endswith('.docx'):
        input_file = os.path.join('./docx_files', filename)
        output_file = os.path.join('./pdf_files', filename.replace('.docx', '.pdf'))
        convert_word_to_pdf(input_file, output_file)
        print(f'Converted {input_file} to {output_file}')