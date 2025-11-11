# OCR Multi-Image Extraction Web App

This web application allows users to upload multiple images of IDs, extract Arabic text fields such as First Name, Family Name, Address, ID, and Date of Birth (DOB) using EasyOCR and YOLO, and optionally download the results as an Excel file.

# Features

- Upload multiple images at once.
- Detect and extract the following fields from ID images:
  - First Name
  - Family Name
  - Address
  - ID number
  - Date of Birth (DOB)
- Real-time extraction results displayed in a table.
- Download results as an Excel file.

## Technology Stack

- **Backend:** Flask
- **OCR & Detection:** EasyOCR, Ultralytics YOLO
- **Frontend:** HTML, CSS, JavaScript (AJAX for dynamic updates)
- **Data Handling:** Pandas, OpenCV
- **Excel Export:** Pandas (XlsxWriter)

---

## Setup Instructions

1.Clone the repository

```bash
git clone https://github.com/yourusername/ocr-multi-image.git
cd ocr-multi-image

2.Install dependencies

pip install -r requirements.txt

3.Run the app

