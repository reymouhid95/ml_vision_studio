from __future__ import annotations

from PIL import Image


def extract_pdf_text(filepath: str) -> str:
    """Extract all text from a PDF using pdfplumber."""
    import pdfplumber
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n\n".join(pages)


def extract_pdf_page_images(filepath: str) -> list[Image.Image]:
    """Render each PDF page to a PIL Image using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    images = []
    doc = fitz.open(filepath)
    mat = fitz.Matrix(1.5, 1.5)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images
