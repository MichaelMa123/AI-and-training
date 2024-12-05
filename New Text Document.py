import PyPDF2

# Path to your PDF file
pdf_path = './86--EIGHTY-SIX Volume-1.pdf'
txt_path = 'output.txt'

# Open the PDF file
with open(pdf_path, 'rb') as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    
    # Extract text from all pages
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        for page in reader.pages:
            text = page.extract_text()
            if text:
                txt_file.write(text)