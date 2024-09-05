# elatentlpips: https://github.com/mingukkang/elatentlpips
# The CC-BY-NC license
# See license file or visit https://github.com/mingukkang/elatentlpips for details

# overfitting_exp.py


from PIL import Image
import os
import argparse

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from diffusers import AutoencoderKL
from elatentlpips.elatentlpips import ELatentLPIPS


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for overfitting exp')
parser.add_argument('--encoder', default='sd3', type=str, help='type of latent encoder to use')


def main(args):
    # Define image loading function
    def load_images(image_folder):
        images = []

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            normalize,
        ])
        for img_name in sorted(os.listdir(image_folder)):
            img_path = os.path.join(image_folder, img_name)
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            images.append(image)
        return torch.stack(images)

    # Define encoding function
    def encode_images(images, vae):
        z = vae.encode(images).latent_dist.sample()
        return z

    def do_overfitting_exp(images, encoder, ensembling, augment, add_l1, num_steps):
        # Initialize the trainable parameter
        in_channels = 4 if encoder in ["sd15", "sd21", "sdxl"] else 16
        trainable_param = torch.nn.Parameter(torch.randn(4, in_channels, 64, 64).to("cuda"), requires_grad=True)

        # Define the reconstruction loss
        elatentlpips = ELatentLPIPS(pretrained=True, net='vgg16', encoder=encoder, augment=augment).to("cuda")
        mse_loss = nn.MSELoss()

        # Define an optimizer
        optimizer = optim.Adam([trainable_param], lr=args.lr)

        # Encode the images
        with torch.no_grad():
            images = images.to("cuda")
            z_images = encode_images(images, vae)

        # Training loop
        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward pass
            loss = elatentlpips(z_images, trainable_param, normalize=True, ensembling=ensembling, add_l1_loss=add_l1).mean()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()


            with torch.no_grad():
                if (step + 1) % 100 == 0:
                    # Calculate ELatentLPIPS loss
                    latent_loss = elatentlpips(z_images, trainable_param, normalize=True, ensembling=False, add_l1_loss=False).mean().item()
                    
                    # Calculate MSE loss
                    decoded_images = vae._decode(trainable_param).sample  # Decoding the trainable parameters
                    mse_loss_value = mse_loss(decoded_images, images).item()

                    # Print the current step and loss values
                    print(f'Step [{step+1}/{num_steps}], LatentLoss: {latent_loss:.4f}, MSE Loss: {mse_loss_value:.4f}')
                    
                    # Append the current loss to the list
                    losses.append(latent_loss)
        
        return losses

    def plot_latentlpips(iterations, lpips_values, labels, colors, encoder):        
        # Create the figure
        plt.figure(figsize=(8, 6))
        
        # Loop over the provided LPIPS values to plot each one
        for lpips, label, color in zip(lpips_values, labels, colors):
            plt.plot(iterations, lpips, label=label, color=color)
        
        # Adding labels and legend
        plt.xlabel('Iteration ($\\times 10^3$)', fontsize=14)
        plt.ylabel('LatentLPIPS', fontsize=14)
        plt.legend(loc='upper right', fontsize=14)
        plt.grid(True)

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.savefig(f'./{encoder}_latentlpips.png')
        
    # Load the pre-trained VAE
    if args.encoder == "sd15":
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    elif args.encoder == "sd21":
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
    elif args.encoder == "sdxl":
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae")
    elif args.encoder == "sd3":
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae")
    elif args.encoder == "flux":
        vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")

    vae = vae.to("cuda", dtype = torch.float32)
    
    # Load the images
    image_folder = './examples'
    images = load_images(image_folder)  # Shape [4, 3, 64, 64]

    num_steps = 15000
    latent_lpips       = do_overfitting_exp(images, args.encoder, False, None, False, num_steps)
    elatent_lpips_b    = do_overfitting_exp(images, args.encoder, True, 'b', False, num_steps)
    elatent_lpips_bg   = do_overfitting_exp(images, args.encoder, True, 'bg', False, num_steps)
    elatent_lpips_bgc  = do_overfitting_exp(images, args.encoder, True, 'bgc', False, num_steps)
    elatent_lpips_bgc0 = do_overfitting_exp(images, args.encoder, True, 'bgco', False, num_steps)

    iterations = np.linspace(0, num_steps, len(latent_lpips))
    lpips_values = [latent_lpips, elatent_lpips_b, elatent_lpips_bg, elatent_lpips_bgc, elatent_lpips_bgc0]
    labels = ['LatentLPIPS', 'E-LatentLPIPS (b)', 'E-LatentLPIPS (bg)', 'E-LatentLPIPS (bgc)', 'E-LatentLPIPS (bgco)']
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    colors = colors[:len(lpips_values)]
    plot_latentlpips(iterations, lpips_values, labels, colors, args.encoder)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)