"""Compute mean PSNR/SSIM using the original 0--255 metric convention."""
import argparse, math
from pathlib import Path
import torch
import torchvision
from pytorch_msssim import ssim

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt',required=True)
    parser.add_argument('--results',required=True)
    args=parser.parse_args()
    gt=Path(args.gt); results=Path(args.results)
    images=sorted(gt.glob('*.png'))
    if not images: raise ValueError('No ground-truth PNG files found')
    psnr_values=[]; ssim_values=[]
    for image in images:
        target=results/image.name
        if not target.is_file(): raise FileNotFoundError(target)
        a=torchvision.io.read_image(str(image)).float().unsqueeze(0)
        b=torchvision.io.read_image(str(target)).float().unsqueeze(0)
        if a.shape!=b.shape: raise ValueError(f'Shape mismatch: {image.name}')
        mse=((a.double()-b.double())/255.).square().mean().item()
        psnr_values.append(float('inf') if mse==0 else -10*math.log10(mse))
        ssim_values.append(ssim(b,a,data_range=255,size_average=True).item())
    print(f'Images: {len(images)}')
    print(f'PSNR: {sum(psnr_values)/len(images):.4f} dB')
    print(f'SSIM: {sum(ssim_values)/len(images):.6f}')

if __name__=='__main__': main()
