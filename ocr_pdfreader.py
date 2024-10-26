import fitz  # PyMuPDF for image handling in PDFs
import pytesseract
from PIL import Image
from io import BytesIO
import os
from PyPDF2 import PdfReader

# Function to extract text from PDF pages, using OCR if the page contains images
def get_pdf_text_with_pages(pdf_paths):
    text_chunks_with_pages = []
    
    for pdf_path in pdf_paths:
        try:
            # Read the PDF for text extraction and image handling
            pdf_reader = PdfReader(pdf_path)
            pdf_document = fitz.open(pdf_path)  # Open PDF with PyMuPDF for image extraction
            doc_name = os.path.basename(pdf_path)
            
            # Iterate through each page
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                # Try to extract text directly from the page
                text = page.extract_text() or ""
                
                # Load page using PyMuPDF
                fitz_page = pdf_document.load_page(page_number - 1)
                images = fitz_page.get_images(full=True)
                
                # Initialize a variable to hold text extracted via OCR
                ocr_text = ""
                
                # If there are images on the page, apply OCR
                for img_index, img in enumerate(images):
                    xref = img[0]  # XREF for the image
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Load the image and apply OCR
                    img_pil = Image.open(BytesIO(image_bytes))
                    ocr_text += pytesseract.image_to_string(img_pil, lang='eng')
                
                # Combine extracted text and OCR text, if available
                combined_text = text + "\n" + ocr_text if ocr_text else text
                
                # Append the text along with page number and document name
                if combined_text.strip():  # Only append if there's text
                    text_chunks_with_pages.append((combined_text, page_number, doc_name))
            
            # Close the PDF document after processing
            pdf_document.close()
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
    
    return text_chunks_with_pages


def main():
    pdf_path = input("Enter the path to your PDF file: ")
    
    # Wrap the pdf_path in a list
    extracted_images = get_pdf_text_with_pages([pdf_path])

    # Print the extracted text with formatting
    for text, page_number, doc_name in extracted_images:
        print(f"Document: {doc_name}, Page: {page_number}\nText:\n{text}\n")


if __name__ == "__main__":
    main()