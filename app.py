from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import easyocr
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# LOAD MODELS 
field_detector = YOLO("models\\new_best.pt")  
digit_detector = YOLO("models\\bestyolo_digit_by_digit.pt")  
reader = easyocr.Reader(['ar'], gpu=True)

# HELPER FUNCTIONS 
def to_arabic(num_str):
    return num_str.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))

def extract_dob_from_id(id_number):
    if len(id_number) != 14 or not id_number.isdigit():
        return "Invalid ID format"
    first = id_number[0]
    year = id_number[1:3]
    month = id_number[3:5]
    day = id_number[5:7]
    full_year = "20" + year if first == "3" else "19" + year
    return f"{full_year}-{month}-{day}"

def easyocr_text_conf_sorted(img, reader, rtl=True):
    try:
        res = reader.readtext(img, detail=1)
    except:
        return ""
    if not res:
        return ""

    word_items = []
    for item in res:
        bbox, txt, prob = item
        if not txt.strip():
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        cx = sum(xs)/len(xs)
        cy = sum(ys)/len(ys)
        word_items.append({"text": txt.strip(), "cx": cx, "cy": cy})

    word_items.sort(key=lambda w: w["cy"])
    lines = []
    line_thresh = 15
    for w in word_items:
        placed = False
        for line in lines:
            if abs(w["cy"] - line["cy"]) <= line_thresh:
                line["words"].append(w)
                line["cy"] = (line["cy"]*(len(line["words"])-1)+w["cy"])/len(line["words"])
                placed = True
                break
        if not placed:
            lines.append({"cy": w["cy"], "words":[w]})

    result_lines = []
    for line in lines:
        line_words = line["words"]
        line_words.sort(key=lambda w: w["cx"], reverse=rtl)
        result_lines.append(" ".join([w["text"] for w in line_words]))

    return " ".join(result_lines).strip()

#  ROUTES 
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_files = request.files.getlist("images")
        if not uploaded_files:
            return {"error": "Please upload at least one image."}, 400

        all_results = []

        for file in uploaded_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Field detection
            results = field_detector(file_path)
            res = results[0]
            target_fields = ["first_name", "family_name", "address"]
            field_texts = {}

            for box in res.boxes:
                cls_name = res.names[int(box.cls)]
                conf = float(box.conf)
                if cls_name in target_fields:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img_rgb[y1:y2, x1:x2]

                    if cls_name == "address" and conf < 0.7:
                        continue

                    if cls_name == "first_name":
                        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        gray = cv2.GaussianBlur(gray, (3,3),0)
                        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,15)
                    else:
                        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

                    extracted_text = easyocr_text_conf_sorted(gray, reader, rtl=True)
                    field_texts[cls_name] = extracted_text

            # ID detection
            digit_results = digit_detector.predict(source=file_path, conf=0.25, save=False, show=False)
            dres = digit_results[0]
            detections = []
            for box in dres.boxes:
                cls_name = dres.names[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, cls_name))
            detections.sort(key=lambda x: x[0])
            detected_id = "".join([cls for _, cls in detections])
            dob = extract_dob_from_id(detected_id)

            all_results.append({
                "first_name": field_texts.get("first_name",""),
                "family_name": field_texts.get("family_name",""),
                "address": field_texts.get("address",""),
                "ID": detected_id,
                "DOB": dob
            })

        # Save Excel for download
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        df = pd.DataFrame(all_results)
        df.to_excel(temp_file.name, index=False)

        return {
            "results": all_results,
            "excel_path": f"/download_excel?path={temp_file.name}"
        }

    return render_template("index.html")

#  Excel Download Route 
@app.route("/download_excel")
def download_excel():
    path = request.args.get('path')
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="ocr_results.xlsx")
    return "File not found", 404

#  RUN APP 
if __name__ == "__main__":
    app.run(debug=True)
