import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_images_root(data_dir: Path) -> Path:
    """
    自動尋找 images 資料夾。
    支援：
    1. data_dir/images/backhand...
    2. data_dir/Tennis Player Actions Dataset.../images/backhand...
    3. data_dir 本身就是包含 class folders 的資料夾
    """
    data_dir = data_dir.resolve()

    possible = [
        data_dir / "images",
        data_dir,
    ]

    for child in data_dir.rglob("images"):
        if child.is_dir():
            possible.append(child)

    expected_classes = {"backhand", "forehand", "ready_position", "serve"}

    for p in possible:
        if not p.exists() or not p.is_dir():
            continue
        child_names = {x.name for x in p.iterdir() if x.is_dir()}
        if len(expected_classes.intersection(child_names)) >= 2:
            return p

    raise FileNotFoundError(
        f"找不到 images 資料夾。請確認 data_dir 是否指到 dataset 根目錄：{data_dir}"
    )


def collect_samples(images_root: Path, selected_classes=None):
    class_dirs = [p for p in images_root.iterdir() if p.is_dir()]

    if selected_classes:
        selected_classes = set(selected_classes)
        class_dirs = [p for p in class_dirs if p.name in selected_classes]

    class_names = sorted([p.name for p in class_dirs])
    if not class_names:
        raise ValueError("沒有找到任何類別資料夾。")

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for class_name in class_names:
        class_dir = images_root / class_name
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMG_EXTS:
                samples.append((str(img_path), class_to_idx[class_name], class_name))

    if not samples:
        raise ValueError("沒有找到任何圖片。")

    return samples, class_names, class_to_idx


class TennisActionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["label"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


def build_transforms(img_size=224, use_hflip=True):
    train_tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.01,
        ),
    ]

    if use_hflip:
        train_tfms.append(transforms.RandomHorizontalFlip(p=0.5))

    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(train_tfms), eval_tfms


def build_model(num_classes: int, freeze_backbone: bool = False):
    """
    MobileNetV3-Small：適合資料量不大、也適合之後部署到一般電腦。
    Torchvision 提供 ImageNet pretrained weights。 
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / total_count
    acc = total_correct / total_count

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    y_true = []
    y_pred = []
    all_probs = []
    all_paths = []

    for images, labels, paths in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_paths.extend(paths)

    avg_loss = total_loss / total_count
    acc = total_correct / total_count

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": all_probs,
        "paths": all_paths,
        "report": report,
        "confusion_matrix": cm,
    }


def save_predictions(eval_result, class_names, out_csv):
    rows = []

    for path, true_idx, pred_idx, probs in zip(
        eval_result["paths"],
        eval_result["y_true"],
        eval_result["y_pred"],
        eval_result["probs"],
    ):
        row = {
            "path": path,
            "true_label": class_names[true_idx],
            "pred_label": class_names[pred_idx],
            "correct": int(true_idx == pred_idx),
        }

        for i, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = probs[i]

        rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


def train(args):
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_root = find_images_root(data_dir)
    print(f"[INFO] images_root = {images_root}")

    selected_classes = args.classes if args.classes else None
    samples, class_names, class_to_idx = collect_samples(images_root, selected_classes)

    print(f"[INFO] classes = {class_names}")
    print(f"[INFO] num_classes = {len(class_names)}")
    print(f"[INFO] total images = {len(samples)}")

    df = pd.DataFrame(samples, columns=["path", "label", "class_name"])

    # stratified split：確保每個類別比例平均
    train_df, temp_df = train_test_split(
        df,
        test_size=args.val_ratio + args.test_ratio,
        stratify=df["label"],
        random_state=args.seed,
    )

    relative_test_ratio = args.test_ratio / (args.val_ratio + args.test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=args.seed,
    )

    print(f"[INFO] train = {len(train_df)}")
    print(f"[INFO] val   = {len(val_df)}")
    print(f"[INFO] test  = {len(test_df)}")

    train_tfms, eval_tfms = build_transforms(
        img_size=args.img_size,
        use_hflip=not args.no_hflip,
    )

    train_set = TennisActionDataset(train_df, transform=train_tfms)
    val_set = TennisActionDataset(val_df, transform=eval_tfms)
    test_set = TennisActionDataset(test_df, transform=eval_tfms)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = get_device()
    print(f"[INFO] device = {device}")

    model = build_model(
        num_classes=len(class_names),
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    if args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location=device)

        if ckpt["class_names"] != class_names:
            raise ValueError(
                f"checkpoint classes {ckpt['class_names']} "
                f"!= current classes {class_names}"
            )

        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] loaded init checkpoint: {args.init_checkpoint}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_result = evaluate(
            model, val_loader, criterion, device, class_names
        )

        val_loss = val_result["loss"]
        val_acc = val_result["acc"]

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={current_lr:.6f}"
        )

        print("[VAL REPORT]")
        print(val_result["report"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            ckpt = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": class_to_idx,
                "img_size": args.img_size,
                "model_name": "mobilenet_v3_small",
            }

            torch.save(ckpt, out_dir / "best_model.pt")
            print(f"[SAVE] best model -> {out_dir / 'best_model.pt'}")

            save_predictions(
                val_result,
                class_names,
                out_dir / "val_predictions.csv",
            )

            pd.DataFrame(
                val_result["confusion_matrix"],
                index=[f"true_{c}" for c in class_names],
                columns=[f"pred_{c}" for c in class_names],
            ).to_csv(out_dir / "val_confusion_matrix.csv", encoding="utf-8-sig")

    pd.DataFrame(history).to_csv(
        out_dir / "history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    # final test using best model
    print("[INFO] Loading best model for final test...")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_result = evaluate(
        model, test_loader, criterion, device, class_names
    )

    print("[TEST REPORT]")
    print(test_result["report"])
    print("[TEST CONFUSION MATRIX]")
    print(test_result["confusion_matrix"])

    save_predictions(
        test_result,
        class_names,
        out_dir / "test_predictions.csv",
    )

    pd.DataFrame(
        test_result["confusion_matrix"],
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    ).to_csv(out_dir / "test_confusion_matrix.csv", encoding="utf-8-sig")

    print(f"[DONE] outputs saved to: {out_dir}")


@torch.no_grad()
def predict(args):
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    model = build_model(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, eval_tfms = build_transforms(img_size=img_size, use_hflip=False)

    img = Image.open(args.image).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]

    print(f"[PRED] image = {args.image}")
    print(f"[PRED] label = {pred_label}")
    print("[PROBS]")
    for class_name, prob in sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True):
        print(f"{class_name}: {prob:.4f}")
        
@torch.no_grad()
def eval_folder(args):
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, folder_class_names, _ = collect_samples(data_dir, selected_classes=None)

    df = pd.DataFrame(samples, columns=["path", "label", "class_name"])

    # 重新用 checkpoint 的 class_names 對 label，避免資料夾排序不同
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    df["label"] = df["class_name"].map(class_to_idx)
    
    print(df[["path", "class_name", "label"]].head(40))
    print(df["class_name"].value_counts())

    _, eval_tfms = build_transforms(img_size=img_size, use_hflip=False)

    dataset = TennisActionDataset(df, transform=eval_tfms)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    result = evaluate(model, loader, criterion, device, class_names)
    
    print("[DEBUG] checkpoint =", args.checkpoint)
    print("[DEBUG] class_names =", class_names)

    print("[EVAL REPORT]")
    print(result["report"])
    print("[CONFUSION MATRIX]")
    print(result["confusion_matrix"])

    save_predictions(
        result,
        class_names,
        out_dir / "my_original_predictions.csv",
    )

    pd.DataFrame(
        result["confusion_matrix"],
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    ).to_csv(out_dir / "my_original_confusion_matrix.csv", encoding="utf-8-sig")

    print(f"[DONE] saved to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")

    train_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Kaggle dataset 解壓縮後的根目錄",
    )

    train_parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/tennis_actions_mobilenetv3",
    )

    train_parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="可選。只訓練指定類別，例如 --classes forehand backhand",
    )

    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--img_size", type=int, default=224)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--label_smoothing", type=float, default=0.05)

    train_parser.add_argument("--val_ratio", type=float, default=0.1)
    train_parser.add_argument("--test_ratio", type=float, default=0.1)

    train_parser.add_argument("--num_workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)

    train_parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="只訓練最後分類層；資料很少時可用，但通常完整 fine-tune 會更好。",
    )

    train_parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="關閉水平翻轉 augmentation。",
    )
    
    train_parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="可選。從既有 checkpoint 繼續 fine-tune。",
    )

    predict_parser = subparsers.add_parser("predict")

    predict_parser.add_argument("--checkpoint", type=str, required=True)
    predict_parser.add_argument("--image", type=str, required=True)
    
    eval_parser = subparsers.add_parser("eval_folder")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--data_dir", type=str, required=True)
    eval_parser.add_argument("--out_dir", type=str, required=True)
    eval_parser.add_argument("--batch_size", type=int, default=32)
    eval_parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "eval_folder":
        eval_folder(args)