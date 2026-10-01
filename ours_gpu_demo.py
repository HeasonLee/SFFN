"""Run SFFN on a grayscale image or folder using a trained checkpoint."""
import argparse
from pathlib import Path
import torch
import torchvision
from convFuse import Net

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    device=torch.device(args.device)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state=checkpoint.get('model_state_dict',checkpoint)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model=Net().to(device)
    model.load_state_dict(state,strict=True)
    model.eval()
    source=Path(args.input)
    images=[source] if source.is_file() else sorted(source.glob('*.png'))
    if not images: raise ValueError('No input images found')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for image in images:
            # Preserve the original training scripts' 0--255 input scale and first-channel convention.
            x=torchvision.io.read_image(str(image))[:1].float().unsqueeze(0).to(device)
            y=model(x).clamp(0,255).squeeze(0).byte().cpu()
            torchvision.io.write_png(y,str(output/(image.stem+'.png')))
            print(image.name)

if __name__=='__main__': main()
