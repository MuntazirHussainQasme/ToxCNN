# ============================================================
# FULL PIPELINE — run top to bottom in ONE Colab session.
# Split at each "# %%" into its own cell.
# Do not let the tab sit idle for long stretches mid-run.
# ============================================================

# %%
# ---- Cell 1: imports + setup ----
import os
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_DIR = Path("/content/bbbc021")
DATA_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def download(url, dest):
    if os.path.exists(dest):
        print(f"Already have {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB", end="")
    print()


# %%
# ---- Cell 2: download metadata CSVs ----
IMAGE_CSV_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv"
COMPOUND_CSV_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_compound.csv"
MOA_CSV_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_moa.csv"

download(IMAGE_CSV_URL, DATA_DIR / "image.csv")
download(COMPOUND_CSV_URL, DATA_DIR / "compound.csv")
download(MOA_CSV_URL, DATA_DIR / "moa.csv")

image_df = pd.read_csv(DATA_DIR / "image.csv")
compound_df = pd.read_csv(DATA_DIR / "compound.csv")
moa_df = pd.read_csv(DATA_DIR / "moa.csv")
print(image_df.shape, compound_df.shape, moa_df.shape)

# %%
# ---- Cell 3: download + extract the 3 image batches ----
IMAGE_ZIP_URLS = [
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24121.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24141.zip",
]

for i, url in enumerate(IMAGE_ZIP_URLS):
    zpath = DATA_DIR / f"batch_{i}.zip"
    download(url, zpath)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(IMG_DIR)
    print(f"Extracted batch {i}")

print("Folders in IMG_DIR:", [p.name for p in IMG_DIR.iterdir()])

# %%
# ---- Cell 4: merge metadata, restrict to downloaded plates ----
image_df = image_df.rename(columns={
    "Image_Metadata_Compound": "compound",
    "Image_Metadata_Concentration": "concentration",
})
labeled_df = image_df.merge(moa_df, on=["compound", "concentration"], how="inner")

downloaded_plates_prefixes = ["Week1_22123", "Week2_24121", "Week2_24141"]
labeled_df["plate_folder"] = labeled_df["Image_Metadata_Plate_DAPI"]
available_df = labeled_df[labeled_df["plate_folder"].isin(downloaded_plates_prefixes)].copy()

print(f"Labeled images available: {len(available_df)}")
print(f"MOA classes covered: {available_df['moa'].nunique()} / 12")
print(available_df["moa"].value_counts())

# %%
# ---- Cell 5: relabel toxic/non-toxic, manual compound split ----
available_df["toxic"] = (available_df["moa"] != "DMSO").astype(int)

dmso_df = available_df[available_df["moa"] == "DMSO"]
treated_df = available_df[available_df["moa"] != "DMSO"]

print(treated_df.groupby("compound").size().sort_values(ascending=False))

# Known-good manual split from prior run (balances by IMAGE count, not just
# compound count — taxol alone is 84 images, so it can't be randomly assigned)
train_compounds = ["taxol", "AZ-A"]
val_compounds = ["ALLN", "cytochalasin B"]
test_compounds = ["cytochalasin D", "methotrexate", "nocodazole"]

train_treated = treated_df[treated_df["compound"].isin(train_compounds)]
val_treated = treated_df[treated_df["compound"].isin(val_compounds)]
test_treated = treated_df[treated_df["compound"].isin(test_compounds)]

dmso_train, dmso_temp = train_test_split(dmso_df, test_size=0.3, random_state=42)
dmso_val, dmso_test = train_test_split(dmso_temp, test_size=0.5, random_state=42)

train_df = pd.concat([train_treated, dmso_train]).reset_index(drop=True)
val_df = pd.concat([val_treated, dmso_val]).reset_index(drop=True)
test_df = pd.concat([test_treated, dmso_test]).reset_index(drop=True)

print(f"Train: {len(train_df)} | toxic={train_df['toxic'].sum()} non-toxic={(train_df['toxic']==0).sum()}")
print(f"Val:   {len(val_df)} | toxic={val_df['toxic'].sum()} non-toxic={(val_df['toxic']==0).sum()}")
print(f"Test:  {len(test_df)} | toxic={test_df['toxic'].sum()} non-toxic={(test_df['toxic']==0).sum()}")

# %%
# ---- Cell 6: build filepaths (note the Week-prefix fix), save CSVs ----
def build_filepath(row, channel="DAPI"):
    fname = row[f"Image_FileName_{channel}"]
    pname = row[f"Image_PathName_{channel}"]
    folder_name = pname.split("/")[-1]  # "Week1/Week1_22123" -> "Week1_22123"
    return str(IMG_DIR / folder_name / fname)

for split_df in [train_df, val_df, test_df]:
    split_df["filepath_dapi"] = split_df.apply(lambda r: build_filepath(r, "DAPI"), axis=1)
    split_df["filepath_tubulin"] = split_df.apply(lambda r: build_filepath(r, "Tubulin"), axis=1)
    split_df["filepath_actin"] = split_df.apply(lambda r: build_filepath(r, "Actin"), axis=1)

for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    exists = split_df["filepath_dapi"].apply(os.path.exists)
    print(f"{name}: {exists.sum()}/{len(split_df)} files found on disk")

train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)
print("Saved train.csv, val.csv, test.csv")

# %%
# ---- Cell 7: image loading + Dataset class ----
def load_composite(row):
    dapi = np.array(Image.open(row["filepath_dapi"]))
    tubulin = np.array(Image.open(row["filepath_tubulin"]))
    actin = np.array(Image.open(row["filepath_actin"]))
    def norm(ch):
        ch = ch.astype(np.float32)
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        ch = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)
        return (ch * 255).astype(np.uint8)
    return np.stack([norm(actin), norm(tubulin), norm(dapi)], axis=-1)

IMG_SIZE = 224
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class BBBC021ToxicDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_composite(row)
        img = self.transform(img)
        return img, row["toxic"]

# %%
# ---- Cell 8: quick visual sanity check ----
import matplotlib.pyplot as plt
sample = train_df.sample(6)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, (_, row) in zip(axes.flat, sample.iterrows()):
    ax.imshow(load_composite(row))
    ax.set_title(f"{row['compound']}\n{'TOXIC' if row['toxic']==1 else 'non-toxic'}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(DATA_DIR / "sample_check.png", dpi=120)
plt.show()

# %%
# ---- Cell 9: single train/val/test baseline (for reference/report) ----
def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in m.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(device)

def run_epoch(loader, model, criterion, optimizer, train_mode):
    model.train() if train_mode else model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.set_grad_enabled(train_mode):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")

train_loader = DataLoader(BBBC021ToxicDataset(train_df, train_tf), batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(BBBC021ToxicDataset(val_df, eval_tf), batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(BBBC021ToxicDataset(test_df, eval_tf), batch_size=16, shuffle=False, num_workers=2)

model = build_model()
class_weights = torch.tensor([len(train_df)/max((train_df['toxic']==0).sum(),1),
                               len(train_df)/max(train_df['toxic'].sum(),1)], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

best_val_f1 = 0
EPOCHS = 15
for epoch in range(EPOCHS):
    tl, ta, tf1 = run_epoch(train_loader, model, criterion, optimizer, True)
    vl, va, vf1 = run_epoch(val_loader, model, criterion, optimizer, False)
    print(f"Epoch {epoch+1}/{EPOCHS} | train acc {ta:.3f} f1 {tf1:.3f} | val acc {va:.3f} f1 {vf1:.3f}")
    if vf1 > best_val_f1:
        best_val_f1 = vf1
        torch.save(model.state_dict(), DATA_DIR / "best_model.pt")

model.load_state_dict(torch.load(DATA_DIR / "best_model.pt"))
tl, ta, tf1 = run_epoch(test_loader, model, criterion, optimizer, False)
print(f"\nSingle-split TEST: acc {ta:.3f} macro-f1 {tf1:.3f}")

# %%
# ---- Cell 10: Leave-one-compound-out cross-validation ----
full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
treated_compounds_all = full_df[full_df["moa"] != "DMSO"]["compound"].unique()
dmso_all = full_df[full_df["moa"] == "DMSO"]
dmso_train_fixed, dmso_test_fixed = train_test_split(dmso_all, test_size=0.3, random_state=42)

print(f"Running LOCO-CV over {len(treated_compounds_all)} compounds:", list(treated_compounds_all))

FOLD_EPOCHS = 3
fold_results = []

for held_out in treated_compounds_all:
    train_treated_fold = full_df[(full_df["moa"] != "DMSO") & (full_df["compound"] != held_out)]
    test_treated_fold = full_df[(full_df["moa"] != "DMSO") & (full_df["compound"] == held_out)]

    fold_train_df = pd.concat([train_treated_fold, dmso_train_fixed]).reset_index(drop=True)
    fold_test_df = pd.concat([test_treated_fold, dmso_test_fixed]).reset_index(drop=True)

    train_loader_f = DataLoader(BBBC021ToxicDataset(fold_train_df, train_tf), batch_size=16, shuffle=True, num_workers=2)
    test_loader_f = DataLoader(BBBC021ToxicDataset(fold_test_df, eval_tf), batch_size=16, shuffle=False, num_workers=2)

    fold_model = build_model()
    n_pos = fold_train_df["toxic"].sum()
    n_neg = len(fold_train_df) - n_pos
    fold_weights = torch.tensor([len(fold_train_df)/max(n_neg,1), len(fold_train_df)/max(n_pos,1)], dtype=torch.float32).to(device)
    fold_criterion = nn.CrossEntropyLoss(weight=fold_weights)
    fold_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fold_model.parameters()), lr=1e-4)

    for epoch in range(FOLD_EPOCHS):
        fold_model.train()
        for imgs, labels in train_loader_f:
            imgs, labels = imgs.to(device), labels.to(device)
            fold_optimizer.zero_grad()
            loss = fold_criterion(fold_model(imgs), labels)
            loss.backward()
            fold_optimizer.step()

    fold_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader_f:
            imgs = imgs.to(device)
            preds = fold_model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    fold_results.append({"held_out": held_out, "n_test": len(fold_test_df), "acc": acc, "f1": f1})
    print(f"Held out '{held_out}' ({len(test_treated_fold)} toxic imgs): acc={acc:.3f} f1={f1:.3f}")

results_df = pd.DataFrame(fold_results)
print("\n=== LOCO-CV Summary ===")
print(results_df)
print(f"Mean accuracy: {results_df['acc'].mean():.3f} +/- {results_df['acc'].std():.3f}")
print(f"Mean macro-F1: {results_df['f1'].mean():.3f} +/- {results_df['f1'].std():.3f}")
results_df.to_csv(DATA_DIR / "loco_cv_results.csv", index=False)
