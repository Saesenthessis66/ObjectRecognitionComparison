import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torchvision

from pytorch_nndct.apis import torch_quantizer
from models.experimental import attempt_load


# ---------------- Dataset ----------------
class ImageDataset(Dataset):
    def __init__(self, img_dir, size=320):
        self.img_dir = img_dir
        self.size = size
        self.files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.files[idx])
        img = read_image(path)

        # Remove alpha channel if present
        if img.shape[0] == 4:
            img = img[:3, :, :]

        img = img.float() / 255.0
        img = torchvision.transforms.Resize((self.size, self.size))(img)
        return img


# ---------------- Quantization ----------------
def run_quant(build_dir, quant_mode, weights, img_dir, img_size):

    print("\n===== QUANTIZATION START =====")
    print(f"Mode       : {quant_mode}")
    print(f"Weights    : {weights}")
    print(f"Calib data : {img_dir}")
    print(f"Output dir : {build_dir}")
    print(f"Image size : {img_size}")

    os.makedirs(build_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    # Load YOLOv5 model
    model = attempt_load(weights)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    quantizer = torch_quantizer(
        quant_mode,
        model,
        (dummy_input,),
        output_dir=build_dir
    )

    quant_model = quantizer.quant_model

    dataset = ImageDataset(img_dir, size=img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("\nRunning calibration/test forward passes...")

    with torch.no_grad():
        for i, imgs in enumerate(loader):
            imgs = imgs.to(device)
            _ = quant_model(imgs)

            if i % 10 == 0:
                print(f"Processed {i}/{len(loader)}")

    # Export
    if quant_mode == 'calib':
        print("\nExporting quant config...")
        quantizer.export_quant_config()

    elif quant_mode == 'test':
        print("\nExporting xmodel...")
        quantizer.export_xmodel(deploy_check=False)

    print("\n===== DONE =====")


# ---------------- CLI ----------------
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 Quantization (Vitis AI)")

    parser.add_argument(
        "--quant_mode",
        type=str,
        default="calib",
        choices=["calib", "test"],
        help="Quantization mode"
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to best.pt"
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Calibration image directory"
    )

    parser.add_argument(
        "--build_dir",
        type=str,
        default="build",
        help="Output directory"
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=320,
        help="Input image size"
    )

    return parser.parse_args()


# ---------------- Main ----------------
if __name__ == "__main__":
    args = parse_args()

    run_quant(
        build_dir=args.build_dir,
        quant_mode=args.quant_mode,
        weights=args.weights,
        img_dir=args.img_dir,
        img_size=args.img_size
    )