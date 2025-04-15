# import torch
# from diffusers import FluxPipeline
#
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
#
# prompt = "A cat holding a sign that says hello world"
# image = pipe(
#     prompt,
#     height=1024,
#     width=1024,
#     guidance_scale=3.5,
#     num_inference_steps=50,
#     max_sequence_length=512,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# image.save("flux-dev.png")


import torch
from diffusers import FluxPipeline
import torch
print(torch.backends.mps.is_available())

# Set device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Load the pipeline with appropriate dtype and move to device
pipe = FluxPipeline.from_pretrained(
"black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32  # use float32 for MPS compatibility
).to(device)

# Optional: remove CPU offload since we already moved it to MPS
# pipe.enable_model_cpu_offload()  # <- comment this out

prompt = "A cat holding a sign that says hello world"

# Use MPS-compatible generator
generator = torch.Generator(device=device).manual_seed(0)

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=generator
).images[0]

image.save("flux-dev.png")
