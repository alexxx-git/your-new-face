from deepface import DeepFace
import random
import time

# тут просто заглушка на выполнение по времени от 2 до 15 секунд
delay = random.uniform(2, 15)
time.sleep(delay)
print(f"we are await about {delay} seconds")
result = DeepFace.analyze(
    img_path="input.jpg", actions=["age"], enforce_detection=False
)

print(f"Возраст: {result[0]['age']} лет")
