from pypdf import PdfReader

reader = PdfReader("Paper91.pdf")
print(f"Total pages: {len(reader.pages)}")