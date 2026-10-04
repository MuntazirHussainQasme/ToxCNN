"""
inference.py — Entry script for ToxCNN
AI for Life Science Challenge submission

Loads a trained ResNet18 toxicity classifier and runs predictions on a
directory of BBBC021-style fluorescence images (DAPI/Tubulin/Actin triplets),
given a metadata CSV with the same structure as BBBC021_v1_image.csv.

Usage:
    python inference.py --model best_model.pt --image_dir ./images --metadata ./test.csv --output predictions.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


IMG_SIZE = 224

eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_composite(filepath_dapi, filepath_tubulin, filepath_actin):
    """Combine the 3 single-channel fluorescence TIFFs into one RGB image."""
    dapi = np.array(Image.open(filepath_dapi))
    tubulin = np.array(Image.open(filepath_tubulin))
    actin = np.array(Image.open(filepath_actin))

    def norm(ch):
        ch = ch.astype(np.float32)
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        ch = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)
        return (ch * 255).astype(np.uint8)

    return np.stack([norm(actin), norm(tubulin), norm(dapi)], axis=-1)


def build_model():
    """Rebuild the ResNet18 architecture used for training (binary head)."""
    m = models.resnet18(weights=None)  # weights loaded from checkpoint, not ImageNet
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def main():
    parser = argparse.ArgumentParser(description="Run ToxCNN toxicity inference")
    parser.add_argument("--model", required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--image_dir", required=True, help="Directory containing the plate image folders")
    parser.add_argument("--metadata", required=True,
                         help="CSV with columns: Image_FileName_DAPI, Image_PathName_DAPI, "
                              "Image_FileName_Tubulin, Image_PathName_Tubulin, "
                              "Image_FileName_Actin, Image_PathName_Actin "
                              "(same format as BBBC021_v1_image.csv)")
    parser.add_argument("--output", default="predictions.csv", help="Where to save predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    df = pd.read_csv(args.metadata)
    image_dir = Path(args.image_dir)
    class_names = ["non-toxic", "toxic"]

    results = []
    with torch.no_grad():
        for idx, row in df.iterrows():
            def resolve(channel):
                pname = row[f"Image_PathName_{channel}"]
                folder = pname.split("/")[-1]
                return image_dir / folder / row[f"Image_FileName_{channel}"]

            fp_dapi, fp_tub, fp_actin = resolve("DAPI"), resolve("Tubulin"), resolve("Actin")

            try:
                img = load_composite(fp_dapi, fp_tub, fp_actin)
            except FileNotFoundError as e:
                print(f"Skipping row {idx}: {e}")
                continue

            img_tensor = eval_tf(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())

            results.append({
                "row": idx,
                "filename_dapi": str(fp_dapi.name),
                "prediction": class_names[pred],
                "prob_non_toxic": float(probs[0]),
                "prob_toxic": float(probs[1]),
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(out_df)} predictions to {args.output}")
    print(out_df["prediction"].value_counts())


if __name__ == "__main__":
    main()
