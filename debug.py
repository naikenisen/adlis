import os
import sys
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from PIL import Image

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importation des fonctions depuis step1
from figures.figure_2_step1 import load_fasterrcnn_model, load_classifier, predict_fasterrcnn, classify_crop

def main():
    det_model_path = os.path.join(project_root, "weights/detection.pth")
    cls_model_path = os.path.join(project_root, "weights/classification.pth")
    test_externe_dir = os.path.join(project_root, "dataset/test_externe")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Chargement des modèles sur {device}...")
    
    # Chargement des modèles
    det_model = load_fasterrcnn_model(det_model_path, device)
    cls_model = load_classifier(cls_model_path, device)
    
    # Récupérer toutes les images du jeu de test externe
    image_files = glob.glob(os.path.join(test_externe_dir, "**", "*.*"), recursive=True)
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"Aucune image trouvée dans {test_externe_dir}")
        return
        
    # Sélectionner 10 images aléatoirement pour avoir un bon aperçu
    selected_images = random.sample(image_files, min(10, len(image_files)))
    
    print("Affichage des 10 images (Fermez la fenêtre d'une image pour passer à la suivante)...")
    
    for img_path in selected_images:
        print(f"Traitement de {os.path.basename(img_path)}...")
        image = Image.open(img_path).convert("RGB")
        
        # Les paramètres optimisés (threshold=0.85, iou=0.50) sont les valeurs par défaut
        boxes = predict_fasterrcnn(image, det_model, device)
        
        fig, ax = plt.subplots(1, figsize=(14, 10))
        ax.imshow(image)
        
        sc_count = 0
        sn_count = 0
        
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            crop = image.crop((xmin, ymin, xmax, ymax))
            
            label = classify_crop(crop, cls_model, device)
            
            # SN en rouge, SC en bleu
            if label == "SC":
                color = "blue"
                sc_count += 1
            else:
                color = "red"
                sn_count += 1
                
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Affichage du label au-dessus de la boîte
            ax.text(xmin, ymin - 8, label, color=color, fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
        plt.title(f"{os.path.basename(img_path)} | SC (Bleu): {sc_count} | SN (Rouge): {sn_count}", 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Affiche l'image en direct (le code met en pause jusqu'à ce que la fenêtre soit fermée)
        plt.show()

if __name__ == "__main__":
    main()
