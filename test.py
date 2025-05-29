from hezar.models import Model
from PIL import Image
import os

# بارگذاری مدل
model = Model.load("hezarai/trocr-base-fa-v1")

# مسیر پوشه تصاویر
image_folder = "DATASET"  # مسیر پوشه تصاویر خود را وارد کنید
output_file = "output.txt"

# باز کردن فایل خروجی برای نوشتن نتایج
with open(output_file, "w", encoding="utf-8") as f:
    for image_name in sorted(os.listdir(image_folder)):
        if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, image_name)
            try:
                # بارگذاری تصویر
                image = Image.open(image_path).convert("RGB")
                # پیش‌بینی متن
                result = model.predict([image_path])
                text = result[0]["text"]
                # نوشتن نتیجه در فایل خروجی
                f.write(f"{image_name} {text}\n")
                print(f"{image_name}: {text}")
            except Exception as e:
                print(f"خطا در پردازش {image_name}: {e}")