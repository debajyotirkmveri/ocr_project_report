import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO

def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)  # Open the PDF with PyMuPDF
    image_list = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]  # XREF for the image
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Load image into PIL for further OCR processing
            img_pil = Image.open(BytesIO(image_bytes))
            image_list.append((page_num, img_index, img_pil))

    pdf_document.close()
    return image_list


def extract_text_from_images(images):
    extracted_text = ""
    
    for page_num, img_index, img in images:
        # Use Tesseract to extract text from image
        text = pytesseract.image_to_string(img)
        extracted_text += f"Page {page_num + 1}, Image {img_index + 1} Text:\n{text}\n"
    
    return extracted_text


def main():
    pdf_path = input("Enter the path to your PDF file: ")

    # Extract images and then text from those images
    extracted_images = extract_images_from_pdf(pdf_path)
    text_from_images = extract_text_from_images(extracted_images)

    print("Text extracted from images:\n", text_from_images)


# Fixing the if statement typo
if __name__ == "__main__":
    main()
