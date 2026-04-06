"""src/gradcam.py  —  Grad-CAM for all three MT-CNN heads + t-SNE"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_lib

_MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
_STD  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)


class GradCAM:
    def __init__(self, model):
        self.model   = model
        self._target = model.backbone_features[-1]
        self._grads  = None
        self._acts   = None
        self._hooks  = [
            self._target.register_forward_hook(
                lambda m,i,o: setattr(self,"_acts",o.detach())),
            self._target.register_full_backward_hook(
                lambda m,gi,go: setattr(self,"_grads",go[0].detach())),
        ]

    def remove_hooks(self):
        for h in self._hooks: h.remove()

    def generate(self, image, task="binary", device="cpu"):
        self.model.eval()
        img = image.to(device)
        self.model.zero_grad()
        out   = self.model(img)
        score = out[task]
        score.sum().backward()
        alpha = self._grads.mean(dim=(2,3), keepdim=True)
        cam   = F.relu((alpha * self._acts).sum(dim=1, keepdim=True))
        cam   = F.interpolate(cam, size=img.shape[-2:],
                              mode="bilinear", align_corners=False)
        cam   = cam.squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        return (cam - lo) / (hi - lo + 1e-8)

    def overlay(self, image_tensor, heatmap, alpha=0.4):
        img = (image_tensor.squeeze(0).cpu() * _STD + _MEAN)
        img = img.permute(1,2,0).numpy().clip(0,1)
        col = cm_lib.jet(heatmap)[...,:3]
        out = (1-alpha)*img + alpha*col
        return (out.clip(0,1)*255).astype(np.uint8)

    @torch.no_grad()
    def visualize_batch(self, loader, device, n=8,
                        save_dir="results/gradcam", tasks=None):
        if tasks is None:
            tasks = ["binary","ordinal1","ordinal2"]
        os.makedirs(save_dir, exist_ok=True)
        images, labs_bin, labs_ord = [], [], []
        for batch in loader:
            for i in range(len(batch["image"])):
                if len(images) >= n: break
                images.append(batch["image"][i:i+1])
                labs_bin.append(int(batch["binary"][i]))
                labs_ord.append(int(batch["ordinal"][i]))
            if len(images) >= n: break

        titles = {"binary":   "Binary Head — P(Demented)",
                  "ordinal1": "Ordinal Head >=1 — P(sev >= Very Mild)",
                  "ordinal2": "Ordinal Head >=2 — P(sev >= Mild/Mod.)"}
        names  = ["Non-Dem.","Very Mild","Mild/Mod."]
        saved  = []
        for task in tasks:
            cols = min(n,4); rows = (n+cols-1)//cols
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols,4*rows))
            axes = np.array(axes).flatten()
            for idx, img_t in enumerate(images):
                img_t = img_t.to(device)
                with torch.enable_grad():
                    hmap = self.generate(img_t, task=task, device=device)
                ov = self.overlay(img_t, hmap)
                axes[idx].imshow(ov); axes[idx].axis("off")
                axes[idx].set_title(
                    f"True: {names[labs_ord[idx]]} (bin={labs_bin[idx]})",
                    fontsize=9)
            for ax in axes[len(images):]: ax.axis("off")
            fig.suptitle(titles[task], fontsize=12)
            plt.tight_layout()
            p = os.path.join(save_dir, f"gradcam_{task}.png")
            fig.savefig(p, bbox_inches="tight", dpi=150); plt.close(fig)
            saved.append(p); print(f"  Saved: {p}")
        return saved


@torch.no_grad()
def tsne_feature_plot(model, loader, device,
                      save_path="results/tsne_features.png"):
    from sklearn.manifold import TSNE
    model.eval()
    feats, labels = [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        feats.append(out["features"].cpu().numpy())
        labels.append(batch["ordinal"].numpy())
    feats  = np.concatenate(feats)
    labels = np.concatenate(labels)
    print(f"  Running t-SNE on {len(feats)} samples ...")
    embed = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(feats)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,6))
    for c, (col, nm) in enumerate(zip(
            ["#2196F3","#FF9800","#F44336"],
            ["Non-Demented","Very Mild","Mild/Moderate"])):
        idx = labels == c
        ax.scatter(embed[idx,0], embed[idx,1], c=col, label=nm,
                   s=15, alpha=0.7, linewidths=0)
    ax.legend(fontsize=10)
    ax.set_title("t-SNE of Shared 256-d Features (MT-CNN)", fontsize=12)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")
