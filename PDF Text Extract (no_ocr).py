import os
import logging
from functools import lru_cache

import fitz  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR
import langdetect

# --- OCR SETTINGS -----------------------------------------------------------

logging.getLogger("ppocr").setLevel(logging.WARNING)

# Folder that contains your PDFs and where temp images will be written
directory = r"D:\VS Code Env\OCR IMPROVED"

# --- PDF split & render -----------------------------------------------------

# --- OCR preloading ---------------------------------------------------------

_OCR_POOL = {}  # lang -> PaddleOCR instance

def preload_ocr_models():
    """Build OCR models once and reuse."""
    # Build the most common languages once. Extend if you need others.
    for lang in ("de", "en"):
        try:
            _OCR_POOL[lang] = get_ocr(lang)  # lru_cache ensures single build per lang
        except Exception:
            pass

def split_pdf(pdf_path: str):
    """Split a PDF into single-page PDFs next to the original file."""
    doc = fitz.open(pdf_path)
    paths = []
    try:
        for page_num in range(len(doc)):
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            output_path = os.path.join(
                directory,
                f"{os.path.basename(pdf_path)[:-4]}_Seite_{page_num + 1}.pdf"
            )
            new_doc.save(output_path)
            new_doc.close()
            paths.append(output_path)
    finally:
        doc.close()
    return paths

def convert_pdf_to_image(pdf_path: str, output_prefix: str, dpi: int = 200):
    """Render a single-page PDF to a PNG image and return the image path."""
    doc = fitz.open(pdf_path)
    image_paths = []
    try:
        page_counter = 1
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            output_path_full = os.path.join(directory, f"{output_prefix}_Seite_{page_counter}.png")
            img.save(output_path_full)
            image_paths.append(output_path_full)
            page_counter += 1
    finally:
        doc.close()
    return image_paths

# --- OCR with cache ---------------------------------------------------------

@lru_cache(maxsize=4)
def get_ocr(lang="de"):
    """Cached PaddleOCR instance."""
    return PaddleOCR(use_textline_orientation=True, lang=lang, det_db_unclip_ratio = 1.5)

def detect_language_from_text(text: str, default: str = "de") -> str:
    """Detect language from already OCR'd text (fallback to default)."""
    text = (text or "").strip()
    if not text:
        return default
    try:
        return langdetect.detect(text) or default
    except Exception:
        return default

# --- Collect OCR tokens per page (texts + confidences) ----------------------

def custom_sort_ocr_results(ocr_result, y_tolerance=40):
    """
    Sortiert OCR-Ergebnisse basierend auf Zeilenhöhe mit Toleranz.
    ocr_result Format pro Item: [ [[x1,y1], ...], ("Text", 0.99) ]
    """
    if not ocr_result:
        return [], []

    # 1. Erstmal alles strikt nach Y-Koordinate (oben links) sortieren
    # x[0][0][1] ist y1 der Bounding Box
    boxes = sorted(ocr_result, key=lambda x: x[0][0][1])

    lines = []
    current_line = [boxes[0]]

    for i in range(1, len(boxes)):
        box = boxes[i]
        current_y = box[0][0][1]
        line_y_ref = current_line[0][0][0][1] # Wir vergleichen mit dem ersten Element der aktuellen Zeile

        # Wenn der Y-Unterschied kleiner als die Toleranz ist -> gleiche Zeile
        if abs(current_y - line_y_ref) < y_tolerance:
            current_line.append(box)
        else:
            # Zeile ist zu Ende -> Jetzt diese Zeile horizontal (X) sortieren
            current_line.sort(key=lambda x: x[0][0][0])
            lines.extend(current_line)
            # Neue Zeile beginnen
            current_line = [box]

    # Letzte Zeile nicht vergessen und auch sortieren
    current_line.sort(key=lambda x: x[0][0][0])
    lines.extend(current_line)

    # Text und Scores extrahieren
    sorted_texts = [item[1][0] for item in lines]
    sorted_scores = [item[1][1] for item in lines]

    return sorted_texts, sorted_scores

def ocr_all_pdfs_to_token_lists(pdf_dir: str):
    """
    Process every PDF in pdf_dir, page-by-page, using prebuilt OCR models.
    """
    all_rec_texts_list = []
    all_rec_scores_list = []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    # Preload models once
    preload_ocr_models()
    ocr_de = _OCR_POOL.get("de") or get_ocr("de")
    ocr_en = _OCR_POOL.get("en") or get_ocr("en")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_slices = split_pdf(pdf_path)

        for pdf_slice in pdf_slices:
            base_prefix = os.path.splitext(os.path.basename(pdf_slice))[0]
            image_paths = convert_pdf_to_image(pdf_slice, output_prefix=f"temp_image_{base_prefix}", dpi=200)

            for image_path in image_paths:
                # Pass 1: DE OCR (fast path)
                results_raw_de = ocr_de.ocr(image_path)

                if results_raw_de and results_raw_de[0]:
                    page_texts, page_scores = custom_sort_ocr_results(results_raw_de[0], y_tolerance=40)
                else:
                    page_texts = []
                    page_scores = []

                # --- SPRACHERKENNUNG & ENGLISCH FALLBACK ---
                joined = " ".join(page_texts)
                detected_lang = detect_language_from_text(joined, default="de")

                if detected_lang.startswith("en"):
                    results_raw_en = ocr_en.ocr(image_path)
                    if results_raw_en and results_raw_en[0]:
                        page_texts, page_scores = custom_sort_ocr_results(results_raw_en[0], y_tolerance=40)

                all_rec_texts_list.append(page_texts)
                all_rec_scores_list.append(page_scores)

                try:
                    os.remove(image_path)
                except Exception:
                    pass

            try:
                os.remove(pdf_slice)
            except Exception:
                pass

    return all_rec_texts_list, all_rec_scores_list

# ============================================================================
# MAIN EXECUTION (OCR ONLY)
# ============================================================================

if __name__ == "__main__":
    preload_ocr_models()  # build models once

    # A) Run OCR over every PDF in 'directory' -> collect tokens & confidences per page
    all_rec_texts_list, all_rec_scores_list = ocr_all_pdfs_to_token_lists(directory)
    
    # OUTPUT VARIABLES:
    # all_rec_texts_list  -> Liste von Listen mit Texten pro Seite
    # all_rec_scores_list -> Liste von Listen mit Scores pro Seite
    print(f"OCR abgeschlossen. Anzahl verarbeiteter Seiten: {len(all_rec_texts_list)}")
    print(all_rec_texts_list)
    print(all_rec_scores_list)