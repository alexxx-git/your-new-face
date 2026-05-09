import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "./models/juggernaut-xl", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

# вход
image = Image.open("face.jpg").convert("RGB")
image = image.resize((1024, 1024))

prompt = "portrait photo of a 60 year old man, realistic skin, wrinkles, high detail"

result = pipe(
    prompt=prompt,
    image=image,
    strength=0.35,
    guidance_scale=6,
    num_inference_steps=30,
).images[0]

result.save("output.png")
