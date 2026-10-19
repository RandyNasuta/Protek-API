import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
import re
from paddleocr import PaddleOCR

ocr_model = None

def get_ocr():
    global ocr_model
    
    if ocr_model is None:
        ocr_model = PaddleOCR(
            lang="id",
            device="cpu",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        
    return ocr_model

STOPWORDS_ID = {
    'yang','dan','di','ke','dari','ini','itu','untuk','pada','dengan','anda','saya',
    'akan','atau','adalah','tidak','juga','ada','kami','kita','dalam','oleh','sebagai',
    'karena','jika','maka','agar','bisa','dapat','telah','sudah','belum','saat','satu',
    'dua','tersebut','yaitu','tetapi','namun','tapi','tak'
}

def clean_text(text):
    text = str(text).lower()
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    text = re.sub(r'http\S+|www\.\S+', ' urltoken ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_image(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.medianBlur(scaled, 3)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY,
        21,
        10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite("debug_hasil_terbaik.png", adaptive_thresh)
    
    return adaptive_thresh

def extract_text_from_image(image_input):
    try:
        processed_image = preprocess_image(image_input)
                
        model = get_ocr()
        results = model.predict(processed_image)
        extracted_text = []
        
        if results is None:
            return ""
        
        for result in results:
            if isinstance(result, dict):
                texts = result.get("rec_texts", [])
            else:
                texts = result.get("rec_texts", [])

                if texts is None:
                    texts = []

            for raw_text in texts:
                cleaned_text = clean_text(raw_text)

                if cleaned_text:
                    extracted_text.append(cleaned_text)
                            
        print("Hasil ekstraksi teks:", extracted_text)
        return " ".join(extracted_text)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error saat ekstrak teks OCR: {e}")