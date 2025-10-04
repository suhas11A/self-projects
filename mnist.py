# mnist_pytorch_onefile.py

import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def build_model():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 32, 3)
            self.c2 = nn.Conv2d(32, 32, 3)
            self.c3 = nn.Conv2d(32, 64, 3)
            self.c4 = nn.Conv2d(64, 64, 3)
            self.fc1 = nn.Linear(64*4*4, 128)
            self.fc2 = nn.Linear(128, 10)
        def forward(self, x):
            x = F.relu(self.c1(x)); x = F.relu(self.c2(x)); x = F.max_pool2d(x, 2); x = F.dropout(x, 0.25, self.training)
            x = F.relu(self.c3(x)); x = F.relu(self.c4(x)); x = F.max_pool2d(x, 2); x = F.dropout(x, 0.25, self.training)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x)); x = F.dropout(x, 0.5, self.training)
            return self.fc2(x)
    return Net()

def get_dataloaders(batch_size, augment):
    t_train = [transforms.ToTensor()]
    if augment:
        t_train = [transforms.RandomRotation(10), transforms.RandomAffine(degrees=0, translate=(0.1,0.1)), transforms.ToTensor()]
    t_val = [transforms.ToTensor()]
    train_full = datasets.MNIST(root=".", train=True, download=True, transform=transforms.Compose(t_train))
    val_size = 6000
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = transforms.Compose(t_val)
    test_ds = datasets.MNIST(root=".", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl, test_dl

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss_sum += criterion(out, yb).item() * yb.size(0)
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return loss_sum/total, correct/total

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(1337); np.random.seed(1337)

    train_dl, val_dl, test_dl = get_dataloaders(args.batch_size, args.augment)
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc, patience_left = 0.0, args.patience
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    ckpt_path = os.path.join(args.output_dir, "best_mnist_cnn.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_loss = loss_sum/total
        train_acc = correct/total

        val_loss, val_acc = evaluate(model, val_dl, device, criterion)
        hist["train_loss"].append(train_loss); hist["val_loss"].append(val_loss)
        hist["train_acc"].append(train_acc);   hist["val_acc"].append(val_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, patience_left = val_acc, args.patience
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping")
                break

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    test_loss, test_acc = evaluate(model, test_dl, device, criterion)
    print(f"Test: acc={test_acc:.4f} loss={test_loss:.4f}")

    y_true, y_pred, wrong_imgs, wrong_true, wrong_pred = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            out = model(xb)
            p = out.argmax(1).cpu()
            y_pred.extend(p.tolist())
            y_true.extend(yb.tolist())
            mism = p != yb
            if mism.any():
                sel = mism.nonzero(as_tuple=False).squeeze(-1)
                for idx in sel:
                    if len(wrong_imgs) >= 25: break
                    wrong_imgs.append(xb[idx].cpu())
                    wrong_true.append(int(yb[idx]))
                    wrong_pred.append(int(p[idx]))
            if len(wrong_imgs) >= 25: break

    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5)); plt.imshow(cm); plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    ticks = np.arange(10); plt.xticks(ticks, ticks); plt.yticks(ticks, ticks); plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png"); plt.savefig(cm_path); plt.close()

    plt.figure(); plt.plot(hist["train_loss"], label="train_loss"); plt.plot(hist["val_loss"], label="val_loss"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout()
    lp = os.path.join(args.output_dir, "loss.png"); plt.savefig(lp); plt.close()
    plt.figure(); plt.plot(hist["train_acc"], label="train_acc"); plt.plot(hist["val_acc"], label="val_acc"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.tight_layout()
    ap = os.path.join(args.output_dir, "acc.png"); plt.savefig(ap); plt.close()

    if wrong_imgs:
        n = int(np.ceil(np.sqrt(len(wrong_imgs))))
        plt.figure(figsize=(8,8))
        for i, img in enumerate(wrong_imgs):
            plt.subplot(n,n,i+1)
            plt.imshow(img.squeeze(0), cmap="gray")
            plt.title(f"T:{wrong_true[i]} P:{wrong_pred[i]}", fontsize=9)
            plt.axis("off")
        plt.tight_layout()
        mp = os.path.join(args.output_dir, "misclassifications.png"); plt.savefig(mp); plt.close()

    final_model = os.path.join(args.output_dir, "mnist_cnn_final.pt")
    torch.save(model.state_dict(), final_model)
    print(f"Saved: {final_model}\nBest ckpt: {ckpt_path}\nArtifacts: {args.output_dir}")

def load_image_tensor(path, invert=None):
    img = Image.open(path).convert("L")
    arr = np.array(img).astype("float32")
    if invert is None:
        invert = arr.mean() > 127.5
    if invert:
        img = ImageOps.invert(img)
    img = img.resize((28,28))
    t = transforms.ToTensor()(img).unsqueeze(0)  # [1,1,28,28], [0,1]
    return t

@torch.no_grad()
def predict(args):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)
    model_path = args.model if args.model else os.path.join(args.output_dir, "mnist_cnn_final.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    x = load_image_tensor(args.predict, invert=args.invert)
    x = x.to(device)
    out = model(x)
    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    print("Prediction:", pred)
    print("Probabilities:", np.round(probs, 4))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--output_dir", type=str, default="artifacts")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--predict", type=str, help="path to digit image")
    p.add_argument("--model", type=str, help=".pt path")
    p.add_argument("--invert", type=lambda x: str(x).lower() in {"1","true","yes"}, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        predict(args)
    else:
        train(args)
