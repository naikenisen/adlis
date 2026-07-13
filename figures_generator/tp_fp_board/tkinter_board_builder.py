import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps, ImageTk
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


MODEL_PATH = "/home/naiken/coding/adlis/Classification_resnet18/best_model_f1.pth"
TESTSET_PATH = "/home/naiken/coding/adlis/classification_dataset_v2/test"

CONFIDENCE_THRESHOLD = 0.70
N_BOARDS_PER_TYPE = 5
CELLS_PER_BOARD = 20
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 640
SEED = 42


class BoardBuilderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Board Builder VP/VN")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_per_type = N_BOARDS_PER_TYPE * CELLS_PER_BOARD

        self.selected_vp = []
        self.selected_vn = []

        self.candidates = []
        self.current_idx = -1
        self.current_photo = None

        self._build_ui()

        try:
            self._prepare_candidates()
            self._next_candidate()
        except Exception as exc:
            messagebox.showerror("Erreur", f"Impossible d'initialiser l'application:\n{exc}")
            self.root.destroy()

    def _build_ui(self):
        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(main, text="Chargement...", anchor="w", justify=tk.LEFT)
        self.info_label.pack(fill=tk.X)

        self.progress_label = tk.Label(main, text="", anchor="w", justify=tk.LEFT)
        self.progress_label.pack(fill=tk.X, pady=(4, 8))

        self.canvas = tk.Label(main, bg="#f0f0f0", width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        button_row = tk.Frame(main)
        button_row.pack(fill=tk.X, pady=(10, 0))

        self.include_button = tk.Button(button_row, text="Inclure (I)", command=self._include_current, width=16)
        self.include_button.pack(side=tk.LEFT)

        self.skip_button = tk.Button(button_row, text="Passer (P)", command=self._skip_current, width=16)
        self.skip_button.pack(side=tk.LEFT, padx=8)

        self.save_button = tk.Button(button_row, text="Sauvegarder maintenant", command=self._save_boards, width=22)
        self.save_button.pack(side=tk.LEFT)

        self.quit_button = tk.Button(button_row, text="Quitter", command=self.root.destroy, width=12)
        self.quit_button.pack(side=tk.RIGHT)

        self.root.bind("<i>", lambda _e: self._include_current())
        self.root.bind("<I>", lambda _e: self._include_current())
        self.root.bind("<p>", lambda _e: self._skip_current())
        self.root.bind("<P>", lambda _e: self._skip_current())
        self.root.bind("<Right>", lambda _e: self._include_current())
        self.root.bind("<Left>", lambda _e: self._skip_current())

    def _prepare_candidates(self):
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_dataset = datasets.ImageFolder(root=TESTSET_PATH, transform=test_transforms)
        num_classes = len(test_dataset.classes)
        if num_classes != 2:
            raise ValueError("Cette application est prevue pour un probleme binaire.")

        model = models.resnet18(weights=None)
        setattr(model, "fc", nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        ))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        y_true, y_score, y_pred = self._evaluate_testset(model, test_dataset)

        df = pd.DataFrame(
            {
                "image_path": [s[0] for s in test_dataset.samples],
                "true_idx": y_true,
                "pred_idx": y_pred,
            }
        )
        df["pred_confidence"] = y_score[np.arange(len(y_pred)), y_pred]

        class_to_idx = test_dataset.class_to_idx
        positive_idx = class_to_idx.get("SC", 0)
        negative_idx = class_to_idx.get("SN", 1 if positive_idx == 0 else 0)

        vp = df[(df["true_idx"] == positive_idx) & (df["pred_idx"] == positive_idx)]
        vn = df[(df["true_idx"] == negative_idx) & (df["pred_idx"] == negative_idx)]

        vp = vp[vp["pred_confidence"] >= CONFIDENCE_THRESHOLD].copy()
        vn = vn[vn["pred_confidence"] >= CONFIDENCE_THRESHOLD].copy()

        vp = vp.sample(frac=1, random_state=SEED).reset_index(drop=True)
        vn = vn.sample(frac=1, random_state=SEED + 1).reset_index(drop=True)

        vp_records = [
            {
                "type": "VP",
                "image_path": rec["image_path"],
                "confidence": rec["pred_confidence"],
            }
            for rec in vp[["image_path", "pred_confidence"]].to_dict("records")
        ]
        vn_records = [
            {
                "type": "VN",
                "image_path": rec["image_path"],
                "confidence": rec["pred_confidence"],
            }
            for rec in vn[["image_path", "pred_confidence"]].to_dict("records")
        ]

        mixed = []
        max_len = max(len(vp_records), len(vn_records))
        for i in range(max_len):
            if i < len(vp_records):
                mixed.append(vp_records[i])
            if i < len(vn_records):
                mixed.append(vn_records[i])

        self.candidates = mixed

        if len(vp_records) < self.target_per_type or len(vn_records) < self.target_per_type:
            messagebox.showwarning(
                "Attention",
                "Le nombre de candidats >=70% est insuffisant pour remplir tous les boards.\n"
                f"VP disponibles: {len(vp_records)} / {self.target_per_type}\n"
                f"VN disponibles: {len(vn_records)} / {self.target_per_type}",
            )

    def _evaluate_testset(self, model, dataset, batch_size=20):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_labels), np.array(all_probs), np.array(all_preds)

    def _next_candidate(self):
        while True:
            self.current_idx += 1

            if self.current_idx >= len(self.candidates):
                self._update_status(final=True)
                self._end_or_wait()
                return

            cand = self.candidates[self.current_idx]
            if cand["type"] == "VP" and len(self.selected_vp) >= self.target_per_type:
                continue
            if cand["type"] == "VN" and len(self.selected_vn) >= self.target_per_type:
                continue

            self._show_candidate(cand)
            self._update_status()
            return

    def _show_candidate(self, cand):
        img = Image.open(cand["image_path"]).convert("RGB")
        img = ImageOps.contain(img, (CANVAS_WIDTH, CANVAS_HEIGHT))
        self.current_photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.current_photo)

        self.info_label.config(
            text=(
                f"Type: {cand['type']} | Confiance: {cand['confidence']:.3f} | "
                f"Image: {cand['image_path']}"
            )
        )

    def _update_status(self, final=False):
        vp_count = len(self.selected_vp)
        vn_count = len(self.selected_vn)
        remaining = max(0, len(self.candidates) - (self.current_idx + 1))

        status = (
            f"Selection VP: {vp_count}/{self.target_per_type} | "
            f"Selection VN: {vn_count}/{self.target_per_type} | "
            f"Candidats restants: {remaining}"
        )
        if final:
            status += "\nFin des candidats disponibles."

        self.progress_label.config(text=status)

    def _include_current(self):
        if self.current_idx < 0 or self.current_idx >= len(self.candidates):
            return

        cand = self.candidates[self.current_idx]
        if cand["type"] == "VP":
            if len(self.selected_vp) < self.target_per_type:
                self.selected_vp.append(cand["image_path"])
        else:
            if len(self.selected_vn) < self.target_per_type:
                self.selected_vn.append(cand["image_path"])

        if self._targets_reached():
            self._save_boards(auto=True)
            return

        self._next_candidate()

    def _skip_current(self):
        self._next_candidate()

    def _targets_reached(self):
        return len(self.selected_vp) >= self.target_per_type and len(self.selected_vn) >= self.target_per_type

    def _save_boards(self, auto=False):
        self._save_type_boards(self.selected_vp, "VP", "final_vp_board")
        self._save_type_boards(self.selected_vn, "VN", "final_vn_board")

        if auto:
            messagebox.showinfo("Termine", "Quota atteint. Boards enregistres automatiquement.")
            self.root.destroy()
        else:
            messagebox.showinfo("Sauvegarde", "Boards sauvegardes.")

    def _save_type_boards(self, image_paths, label, out_prefix):
        if not image_paths:
            return

        for board_idx in range(N_BOARDS_PER_TYPE):
            start = board_idx * CELLS_PER_BOARD
            end = start + CELLS_PER_BOARD
            chunk = image_paths[start:end]
            if not chunk:
                continue

            out_path = f"{out_prefix}_{board_idx + 1}.png"
            title = f"{label} - board {board_idx + 1}"
            self._plot_grid(chunk, title, out_path)

    def _plot_grid(self, image_paths, title, out_path):
        n = len(image_paths)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))

        axes = np.array(axes).reshape(-1)

        for i, path in enumerate(image_paths):
            img = plt.imread(path)
            axes[i].imshow(img)
            axes[i].axis("off")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=10, fontweight="bold")
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def _end_or_wait(self):
        if self._targets_reached():
            self._save_boards(auto=True)
            return

        missing_vp = self.target_per_type - len(self.selected_vp)
        missing_vn = self.target_per_type - len(self.selected_vn)
        messagebox.showinfo(
            "Fin des candidats",
            "Plus de candidats disponibles.\n"
            f"VP manquants: {max(0, missing_vp)}\n"
            f"VN manquants: {max(0, missing_vn)}\n"
            "Tu peux sauvegarder ce qui a ete selectionne avec 'Sauvegarder maintenant'.",
        )


if __name__ == "__main__":
    app_root = tk.Tk()
    BoardBuilderApp(app_root)
    app_root.mainloop()
