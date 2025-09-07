import pytesseract
import cv2
import numpy as np
import requests
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_url = "https://upload.wikimedia.org/wikipedia/commons/8/88/2019_Toyota_Camry_XSE_%28GSV70R%29_sedan_%282018-10-12%29_01.jpg"
download_path = "images/vehicle_sample.jpg"
output_text_path = "output/extracted_text.txt"
damage_output_path = "output/damage_detected.jpg"

os.makedirs("images", exist_ok=True)
os.makedirs("output", exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(image_url, headers=headers)

if response.status_code != 200:
    print(f"❌ Failed to download image from URL. Status Code: {response.status_code}")
    exit()

with open(download_path, "wb") as f:
    f.write(response.content)

image = cv2.imread(download_path)
if image is None:
    print("❌ Error: Unable to read downloaded image")
    exit()

print("\n🔍 Step 1: Extracting vehicle info from image...\n")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray)

print("📄 Extracted Vehicle Text:\n")
print(text)

with open(output_text_path, "w") as f:
    f.write(text)
print(f"\n✅ Extracted text saved to: {output_text_path}")

print("\n🔧 Step 2: Analyzing for visible damage...\n")
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
damage_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
damage_mask[:, :, 1] = 0
damage_mask[:, :, 0] = 0

result = cv2.addWeighted(image, 0.8, damage_mask, 1, 0)
cv2.imwrite(damage_output_path, result)

print(f"✅ Damage-highlighted image saved to: {damage_output_path}")
print("\n🎉 All tasks completed successfully!\n")
