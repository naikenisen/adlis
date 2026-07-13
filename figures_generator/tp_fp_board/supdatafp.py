import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Paths
MODEL_PATH = "/home/naiken/coding/adlis/Classification_resnet18/best_model_f1.pth"
TESTSET_PATH = "/home/naiken/coding/adlis/classification_dataset_v2/test"
IMAGES_PER_BOARD = 20
CONFIDENCE_THRESHOLD = 0.2
THUMBNAIL_SIZE = (160, 160)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (same as validation)
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = datasets.ImageFolder(root=TESTSET_PATH, transform=test_transforms)
num_classes = len(test_dataset.classes)

# Model (same architecture as training)
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
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
    nn.Linear(64, num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Get class names and mapping
class_to_idx = test_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Map technical labels to descriptive names
label_mapping = {
    'SC': 'ring sideroblasts',
    'SN': 'erythroblasts'
}


def get_image_paths(dataset):
    return [sample[0] for sample in dataset.samples]


def evaluate_testset(dataset, batch_size=20):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds),
    )


y_true, y_score, y_pred = evaluate_testset(test_dataset)
image_paths = get_image_paths(test_dataset)

# Build DataFrame with results
df = pd.DataFrame({
    'image_path': image_paths,
    'true_idx': y_true,
    'pred_idx': y_pred,
})
df['pred_confidence'] = y_score[np.arange(len(y_pred)), y_pred]
df['true_label'] = df['true_idx'].map(idx_to_class)
df['pred_label'] = df['pred_idx'].map(idx_to_class)
df['true_label_desc'] = df['true_label'].map(label_mapping)
df['pred_label_desc'] = df['pred_label'].map(label_mapping)

# Identify FP using class names when available
if num_classes != 2:
    raise ValueError("Ce script est prevu pour une classification binaire (2 classes).")

positive_idx = class_to_idx.get('SC', 0)

FP = df[(df['true_idx'] != positive_idx) & (df['pred_idx'] == positive_idx)]

# Keep only false positives above confidence threshold
FP = FP[FP['pred_confidence'] >= CONFIDENCE_THRESHOLD]


class ImageSelectorApp:
    def __init__(self, root, fp_df, images_per_page=20, ncols=4):
        self.root = root
        self.root.title("Selection FP - ADLIS")
        self.root.geometry("900x780")
        self.fp_df = fp_df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.images_per_page = images_per_page
        self.ncols = ncols
        self.current_page = 0
        self.photo_refs = []

        self.selected_paths = set()
        self.checkbox_vars = {}

        self._build_ui()
        self._render_page()

        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_ui(self):
        header = (
            f"Faux positifs disponibles: {len(self.fp_df)} | "
            f"Confiance min: {CONFIDENCE_THRESHOLD:.2f}"
        )
        self.header_label = tk.Label(self.root, text=header, font=("Arial", 11, "bold"))
        self.header_label.pack(pady=8)

        self.page_info = tk.Label(self.root, text="", font=("Arial", 10))
        self.page_info.pack(pady=(0, 8))

        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        self.canvas = tk.Canvas(self.content_frame, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.content_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.grid_frame = tk.Frame(self.canvas)

        self.grid_frame.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self._resize_canvas_window)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        controls_frame = tk.Frame(self.root)
        controls_frame.pack(fill=tk.X, padx=12, pady=8)

        self.prev_btn = tk.Button(controls_frame, text="<< Precedent", command=self.prev_page)
        self.prev_btn.pack(side=tk.LEFT)

        self.next_btn = tk.Button(controls_frame, text="Suivant >>", command=self.next_page)
        self.next_btn.pack(side=tk.LEFT, padx=6)

        self.selected_label = tk.Label(controls_frame, text="Selectionnes: 0", font=("Arial", 10, "bold"))
        self.selected_label.pack(side=tk.LEFT, padx=10)

        self.save_btn = tk.Button(
            controls_frame,
            text="Sauvegarder la selection",
            command=self.save_selection
        )
        self.save_btn.pack(side=tk.RIGHT)

    def _resize_canvas_window(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _render_page(self):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.photo_refs = []
        self.checkbox_vars = {}

        total_images = len(self.fp_df)
        if total_images == 0:
            self.page_info.config(text="Aucune image FP trouvee.")
            return

        start = self.current_page * self.images_per_page
        end = min(start + self.images_per_page, total_images)
        page_df = self.fp_df.iloc[start:end]

        total_pages = int(np.ceil(total_images / self.images_per_page))
        self.page_info.config(
            text=f"Page {self.current_page + 1}/{total_pages} | Images {start + 1}-{end}"
        )

        for i, (_, row) in enumerate(page_df.iterrows()):
            card = tk.Frame(self.grid_frame, relief=tk.RIDGE, bd=1)
            r, c = divmod(i, self.ncols)
            card.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")

            img = Image.open(row['image_path']).convert("RGB")
            img.thumbnail(THUMBNAIL_SIZE)
            photo = ImageTk.PhotoImage(img)
            self.photo_refs.append(photo)

            img_label = tk.Label(card, image=photo)
            img_label.pack(padx=4, pady=4)

            conf_text = f"Conf: {row['pred_confidence']:.2f}"
            tk.Label(card, text=conf_text, font=("Arial", 9)).pack(pady=(0, 2))

            path = row['image_path']
            var = tk.BooleanVar(value=path in self.selected_paths)
            self.checkbox_vars[path] = var

            cb = tk.Checkbutton(
                card,
                text="Selectionner",
                variable=var,
                command=lambda p=path: self.toggle_selection(p)
            )
            cb.pack(pady=(0, 4))

        self.prev_btn.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if end < total_images else tk.DISABLED)
        self._refresh_selected_label()
        self.canvas.yview_moveto(0)

    def toggle_selection(self, path):
        var = self.checkbox_vars[path]
        if var.get():
            self.selected_paths.add(path)
        else:
            self.selected_paths.discard(path)
        self._refresh_selected_label()

    def _refresh_selected_label(self):
        self.selected_label.config(text=f"Selectionnes: {len(self.selected_paths)}")

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._render_page()

    def next_page(self):
        total_pages = int(np.ceil(len(self.fp_df) / self.images_per_page))
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self._render_page()

    def save_selection(self):
        if not self.selected_paths:
            messagebox.showwarning("Selection vide", "Aucune image selectionnee.")
            return

        out_file = filedialog.asksaveasfilename(
            title="Enregistrer la liste des images selectionnees",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("CSV file", "*.csv"), ("All files", "*.*")]
        )
        if not out_file:
            return

        selected_df = self.fp_df[self.fp_df['image_path'].isin(self.selected_paths)].copy()
        selected_df = selected_df.sort_values(by='pred_confidence', ascending=False)

        if out_file.lower().endswith(".csv"):
            selected_df.to_csv(out_file, index=False)
        else:
            with open(out_file, "w", encoding="utf-8") as file:
                for path in selected_df['image_path']:
                    file.write(f"{path}\n")

        messagebox.showinfo(
            "Sauvegarde terminee",
            f"{len(selected_df)} image(s) enregistree(s) dans:\n{out_file}"
        )


if __name__ == "__main__":
    if FP.empty:
        print("Aucune image FP ne passe le seuil de confiance.")
    else:
        app_root = tk.Tk()
        app = ImageSelectorApp(app_root, FP, images_per_page=IMAGES_PER_BOARD)
        app_root.mainloop()