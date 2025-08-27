import requests
import json
import base64

URL = "http://127.0.0.1:8000/predict"
IMG = r"D:\Data-Science-Camp-Tasks\Home_Task_30\photo_2025-08-27_22-33-16.jpg"

with open(IMG, "rb") as f:
    # ВАЖЛИВО: явно вказуємо content-type
    files = {"file": ("photo.jpg", f, "image/jpeg")}
    params = {"return_image": "true"}  # або "false"
    resp = requests.post(URL, files=files, params=params, timeout=120)

print(resp.status_code, resp.reason)
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))

# Якщо просили return_image=true — збережемо анотоване PNG
data = resp.json()
b64 = data.get("annotated_image_base64")
if b64:
    with open("annotated.png", "wb") as out:
        out.write(base64.b64decode(b64))
    print("Saved annotated.png")
