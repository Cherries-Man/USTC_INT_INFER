from engine import Engine
import os

weight_path = os.getenv("HF_MODELS_CACHE")
if weight_path is None:
    raise ValueError("HF_MODELS_CACHE environment variable is not set")

engine = Engine(weight_path)
batch = ["Hello how is it going?", "Is this a cat?"]
result = engine.execute(batch)

print(result)
