# from deepface import DeepFace
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
import torch
from PIL import Image
from pathlib import Path


device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# print(f"PyTorch: {torch.__file__}")
# print(f"CUDA: {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
# print(f"PyTorch version: {torch.__version__}")
# --- 1. Получаем лицо ---
# faces = DeepFace.extract_faces("new_face/input.jpg", enforce_detection=False)

# if len(faces) == 0:
#     raise ValueError("No face")

# face = faces[0]["face"]  # RGB float [0..1]
# face = (face * 255).astype(np.uint8)

# # numpy → PIL
# init_image = Image.fromarray(face).resize((512, 512))
# --- 2. Загружаем SDXL ---

# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
# ).to("cuda")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float16
).to("cuda")


# # (ускорение и экономия VRAM ~10GB)
pipe.enable_attention_slicing()  # экономит память при генерации
# pipe.enable_model_cpu_offload()  # можно перевести часть модели на CPU

# --- 3. Prompt---
prompt = "portrait of a 65 year old person, realistic skin, wrinkles, detailed face, high quality"
init_image = ""
# --- 4. Генерация ---
result = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.4,  # меньше = лучше сохраняется лицо
    guidance_scale=2.0,
).images[0]

# --- 5. Сохранение ---
combined_width = init_image.width + result.width
combined_height = max(init_image.height, result.height)
combined = Image.new("RGB", (combined_width, combined_height))
combined.paste(init_image, (0, 0))
combined.paste(result, (init_image.width, 0))

# --- 6. Сохраняем ---
output_path = Path("output") / "comparison_result.jpg"
output_path.parent.mkdir(parents=True, exist_ok=True)
combined.save(output_path)
print("Готово: comparison_result.jpg")
