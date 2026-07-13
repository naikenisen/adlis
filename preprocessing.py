"""
    Ce scripte génère le dossier "classification_set" et le CSV "split". 
    Il utilise : @dataset/metadata.csv et @dataset/images et @dataset/annotations. 
    Il crée dans @dataset/ le CSV "split.csv" qui comporte 2 colonnes : 
    filename et split. On rempli la colonne split avec les labels: train, valid, test, 
    découpé respectivement en 70, 15, 15. 
    Le fichier @dataset/metadata.csv sert à répartir les labels selon les filename. 
    Sachant que chaque patient correspond à la colonne "directory_name", il ne faut 
    jamais qu'une image d'une patient se retrouves dans un même set (train, valid, test).
    D'autre part, il faut respecter une répartition la plus homogène possible des 
    sidero_count selon : 
            sidero_count <5
            sidero_count between 5 à 15%
            sidero_count > 15%
    répartit de manière homogène selon train, valid, test. 

    Une fois @dataset/split.csv créé, le scripte crée un dossier nommé 
    @dataset/classification_set avec les sous-dossiers (train, valid, test). La suite du travail
    reprendra les labels de la colonne split de split.csv pour répartir les images selon les sous 
    dossiers. 
    Il découpe des sous-images depuis les images du dossier @dataset/images en utilisant 
    les coordonées de @dataset/annotations correspondantes et nommes ces sous-images selon 
    la balise <name> des anotations. Ainsi, les images auront soit le nom SN ou SC sachant
    que ce sont les deux valeurs que peuvent prendre la balise <name> des anotations. Ainsi, 
    le scripte créera aussi les sous dossiers respectifs:
     - @dataset/classification_set/train/SC
     - @dataset/classification_set/train/SN
     - @dataset/classification_set/valid/SC
     - @dataset/classification_set/valid/SN
     - @dataset/classification_set/test/SC
     - @dataset/classification_set/test/SN
"""

import os
import csv
import random
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

def main():
    metadata_path = 'dataset/metadata.csv'
    images_dir = 'dataset/images'
    annotations_dir = 'dataset/annotations'
    split_csv_path = 'dataset/split.csv'
    class_set_dir = 'dataset/classification_set'

    # 1. Lire metadata.csv et regrouper par patient (directory_name)
    patients = {}  # dict: directory_name -> {'category': str, 'filenames': list}
    
    if not os.path.exists(metadata_path):
        print(f"Erreur : le fichier {metadata_path} n'existe pas.")
        return

    print("Lecture de metadata.csv et regroupement par patient...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            directory_name = row.get('directory_name', '')
            filename = row.get('filename', '')
            sidero_count_str = row.get('sidero_count', '').replace('%', '').replace(',', '.').strip()
            
            if not directory_name or not filename:
                continue
                
            try:
                sidero_count = float(sidero_count_str)
            except ValueError:
                # Si la valeur est vide ou non numérique, on l'ignore
                continue
                
            # Détermination de la catégorie
            if sidero_count < 5:
                category = '<5'
            elif 5 <= sidero_count <= 15:
                category = '5-15'
            else:
                category = '>15'
                
            if directory_name not in patients:
                patients[directory_name] = {'category': category, 'filenames': []}
            
            if filename not in patients[directory_name]['filenames']:
                patients[directory_name]['filenames'].append(filename)

    # 2. Répartition homogène des patients (train: 70%, valid: 15%, test: 15%)
    # On fixe une seed pour que le split soit reproductible d'une exécution à l'autre
    random.seed(42)
    split_assignments = {}  # dict: filename -> split ('train', 'valid', 'test')
    
    categories = ['<5', '5-15', '>15']
    for cat in categories:
        cat_patients = [p for p, data in patients.items() if data['category'] == cat]
        # Mélanger aléatoirement les patients de cette catégorie
        random.shuffle(cat_patients)
        
        n = len(cat_patients)
        n_train = int(0.70 * n)
        n_valid = int(0.15 * n)
        
        train_patients = cat_patients[:n_train]
        valid_patients = cat_patients[n_train:n_train+n_valid]
        test_patients = cat_patients[n_train+n_valid:]
        
        for p in train_patients:
            for f in patients[p]['filenames']:
                split_assignments[f] = 'train'
        for p in valid_patients:
            for f in patients[p]['filenames']:
                split_assignments[f] = 'valid'
        for p in test_patients:
            for f in patients[p]['filenames']:
                split_assignments[f] = 'test'
                
    # 3. Création du fichier dataset/split.csv
    print(f"Création du fichier {split_csv_path}...")
    with open(split_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'split'])
        for filename, split in split_assignments.items():
            writer.writerow([filename, split])
            
    # 4. Création de l'arborescence dataset/classification_set/
    print(f"Création de l'arborescence dans {class_set_dir}...")
    splits = ['train', 'valid', 'test']
    classes = ['SN', 'SC']
    
    for s in splits:
        for c in classes:
            os.makedirs(os.path.join(class_set_dir, s, c), exist_ok=True)
            
    # 5. Découpage des images et enregistrement dans les bons dossiers
    print("Découpage des sous-images en cours... (Cette opération peut prendre du temps)")
    
    images_processed = 0
    sub_images_created = 0
    
    for filename, split in tqdm(split_assignments.items(), desc="Découpage des images"):
        img_path = os.path.join(images_dir, filename)
        anno_filename = os.path.splitext(filename)[0] + '.xml'
        anno_path = os.path.join(annotations_dir, anno_filename)
        
        if not os.path.exists(img_path):
            print(f"Image introuvable : {img_path}")
            continue
            
        if not os.path.exists(anno_path):
            # Certaines images pourraient ne pas avoir d'annotations
            continue
            
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Erreur d'ouverture de l'image {img_path}: {e}")
            continue
            
        try:
            tree = ET.parse(anno_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Erreur de parsing de l'annotation {anno_path}: {e}")
            continue
            
        obj_idx = 0
        file_sub_images_created = False
        
        for obj in root.iter('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            name = name_elem.text.strip()
            
            # Ne garder que les classes connues ('SN' et 'SC')
            if name not in classes:
                continue
                
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
                
            try:
                # Les coordonnées XML peuvent être des flottants convertis en string
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                
                # Découpage avec PIL
                crop_img = img.crop((xmin, ymin, xmax, ymax))
                
                # Nommage: nom d'origine _ index.jpg
                out_name = f"{os.path.splitext(filename)[0]}_{obj_idx}.jpg"
                out_path = os.path.join(class_set_dir, split, name, out_name)
                
                # Sauvegarde de la sous-image
                crop_img.save(out_path)
                obj_idx += 1
                sub_images_created += 1
                file_sub_images_created = True
                
            except Exception as e:
                print(f"Erreur lors du découpage d'un objet dans {filename}: {e}")
                
        if file_sub_images_created:
            images_processed += 1
            
    print("\nTerminé !")
    print(f"Fichier CSV de répartition créé : {split_csv_path}")
    print(f"Images originales ayant servi au découpage : {images_processed}")
    print(f"Total de sous-images générées : {sub_images_created}")

if __name__ == "__main__":
    main()
