#create dataset
import os
import shutil
import random

# Chemins source
DATASET_DIR = '/home/naiken/coding/ADLIS/Object_detection_fast_rcnn/object_detection_dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
ANNOT_DIR = os.path.join(DATASET_DIR, 'annotations')

# Chemins destination
SPLITS = ['train', 'valid', 'test']
SPLIT_RATIOS = [0.7, 0.15, 0.15]  # 70% train, 15% valid, 15% test

DEST_DIR = os.path.join(os.getcwd(), 'dataset_split')
for split in SPLITS:
    os.makedirs(os.path.join(DEST_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, split, 'annotations'), exist_ok=True)

# Récupérer toutes les images
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm'))]
random.shuffle(image_files)

n_total = len(image_files)
n_train = int(n_total * SPLIT_RATIOS[0])
n_valid = int(n_total * SPLIT_RATIOS[1])
n_test = n_total - n_train - n_valid

splits = {
    'train': image_files[:n_train],
    'valid': image_files[n_train:n_train+n_valid],
    'test': image_files[n_train+n_valid:]
}

for split, files in splits.items():
    for img_file in files:
        # Copier image
        src_img = os.path.join(IMAGES_DIR, img_file)
        dst_img = os.path.join(DEST_DIR, split, 'images', img_file)
        shutil.copy2(src_img, dst_img)
        # Copier annotation
        annot_file = img_file + '.xml' if not img_file.endswith('.xml') else img_file
        annot_file = os.path.splitext(img_file)[0] + '.jpg.xml'  # Format utilisé dans ton dataset
        src_annot = os.path.join(ANNOT_DIR, annot_file)
        dst_annot = os.path.join(DEST_DIR, split, 'annotations', annot_file)
        if os.path.exists(src_annot):
            shutil.copy2(src_annot, dst_annot)
        else:
            print(f"Annotation manquante pour {img_file}")

print("Split terminé !")
print(f"Train: {len(splits['train'])}, Valid: {len(splits['valid'])}, Test: {len(splits['test'])}")
