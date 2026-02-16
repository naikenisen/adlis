import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
# Chemin du dataset d'origine
DATASET_DIR = 'object_detection_dataset/dataset_split'
OUTPUT_DIR = 'classification_dataset_v2'
SPLITS = ['train', 'test', 'valid']
CLASSES = ['SC', 'SN']

def ensure_dirs():
	for split in SPLITS:
		for cls in CLASSES:
			out_dir = os.path.join(OUTPUT_DIR, split, cls)
			os.makedirs(out_dir, exist_ok=True)

def parse_xml(xml_path):
	tree = ET.parse(xml_path)
	root = tree.getroot()
	objects = []
	filename = root.find('filename').text
	for obj in root.findall('object'):
		name = obj.find('name').text
		bbox = obj.find('bndbox')
		xmin = int(bbox.find('xmin').text)
		ymin = int(bbox.find('ymin').text)
		xmax = int(bbox.find('xmax').text)
		ymax = int(bbox.find('ymax').text)
		objects.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})
	return filename, objects

def process_split(split):
	ann_dir = os.path.join(DATASET_DIR, split, 'annotations')
	img_dir = os.path.join(DATASET_DIR, split, 'images')
	xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
	batch_size = 100
	total = len(xml_files)
	for batch_start in tqdm(range(0, total, batch_size), desc=f"{split} annotations (batch)"):
		batch_files = xml_files[batch_start:batch_start+batch_size]
		for xml_file in batch_files:
			xml_path = os.path.join(ann_dir, xml_file)
			filename, objects = parse_xml(xml_path)
			img_path = os.path.join(img_dir, filename)
			if not os.path.exists(img_path):
				print(f"Image not found: {img_path}")
				continue
			try:
				img = Image.open(img_path)
			except Exception as e:
				print(f"Erreur ouverture image {img_path}: {e}")
				continue
			for idx, obj in enumerate(objects):
				cls = obj['name']
				if cls not in CLASSES:
					continue
				xmin, ymin, xmax, ymax = obj['bbox']
				crop = img.crop((xmin, ymin, xmax, ymax))
				new_name = f"{os.path.splitext(filename)[0]}_{cls}_{idx}.jpg"
				out_dir = os.path.join(OUTPUT_DIR, split, cls)
				out_path = os.path.join(out_dir, new_name)
				crop.save(out_path)
			img.close()

def main():
	ensure_dirs()
	for split in SPLITS:
		process_split(split)
	print("Découpage terminé.")

if __name__ == '__main__':
	main()
