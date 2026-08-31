import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from gabor_noise_renderer import render, render_with_randomizer_color, GaborModel


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
print("device: ", device)

if len(sys.argv) < 2:
    print("usage: python gabor_noise_reproduce.py <path_to_model_weights.pt>")
    sys.exit(1)

weights_path = sys.argv[1]
output_dir = os.path.dirname(weights_path)

model, kernel_size = GaborModel.load_weights(weights_path)

with torch.no_grad():
    rendered_image = render(kernel_size=kernel_size, model=model, device=device)

output_img_array = rendered_image.cpu().detach().numpy()
output_path = os.path.join(output_dir, 'reproduced_image.png')
plt.imsave(output_path, output_img_array)
print(f"reproduced image saved to {output_path}")

with torch.no_grad():
    rendered_image = render_with_randomizer_color(kernel_size=kernel_size, model=model, device=device)

output_img_array = rendered_image.cpu().detach().numpy()
output_path = os.path.join(output_dir, 'reproduced_randomized_image.png')
plt.imsave(output_path, output_img_array)
print(f"randomized color image saved to {output_path}")
