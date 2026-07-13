import os
import csv
import statistics
import matplotlib.pyplot as plt

def main():
    patients = {}
    with open('dataset/metadata.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            d_name, f_name = row.get('directory_name', ''), row.get('filename', '')
            if not d_name or not f_name: continue
            try:
                s_count = float(row.get('sidero_count', '').replace('%', '').replace(',', '.').strip())
            except ValueError: continue
            
            if d_name not in patients:
                patients[d_name] = {'sidero': s_count, 'files': [], 'type_lame': []}
            if f_name not in patients[d_name]['files']:
                patients[d_name]['files'].append(f_name)
                patients[d_name]['type_lame'].append(row.get('type_lame', '').strip().lower())

    n_patients = len(patients)
    slides = [len(p['files']) for p in patients.values()]
    m_slides = statistics.mean(slides) if slides else 0
    min_s, max_s = (min(slides), max(slides)) if slides else (0, 0)
    sd_slides = statistics.stdev(slides) if len(slides) > 1 else 0

    def get_cat(s): return '<5%' if s < 5 else ('5-15%' if s <= 15 else '>15%')
    
    cat_counts = {'<5%': 0, '5-15%': 0, '>15%': 0}
    type_counts = {'frottis': 0, 'ecrasement': 0}
    
    for p in patients.values():
        cat_counts[get_cat(p['sidero'])] += 1
        for tl in p['type_lame']:
            if 'frottis' in tl: type_counts['frottis'] += 1
            elif 'ecrasement' in tl: type_counts['ecrasement'] += 1

    file_to_split = {}
    if os.path.exists('dataset/split.csv'):
        with open('dataset/split.csv', 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                file_to_split[row['filename']] = row['split']

    split_stats = {s: {'<5%': 0, '5-15%': 0, '>15%': 0} for s in ['train', 'valid', 'test']}
    for d_name, data in patients.items():
        if data['files']:
            split = file_to_split.get(data['files'][0])
            if split in split_stats:
                split_stats[split][get_cat(data['sidero'])] += 1

    class_stats = {s: {'SC': 0, 'SN': 0} for s in ['train', 'valid', 'test']}
    for s in class_stats:
        for c in class_stats[s]:
            d = os.path.join('dataset/classification_set', s, c)
            if os.path.exists(d):
                class_stats[s][c] = len([name for name in os.listdir(d) if os.path.isfile(os.path.join(d, name))])

    data = [
        ["Variable", "Value"],
        ["Metadata (General)", ""],
        ["Total Patients (n)", str(n_patients)],
        ["Slides per patient (mean ± SD [min; max])", f"{m_slides:.2f} ± {sd_slides:.2f} [{min_s}; {max_s}]"],
        ["Sidero Count < 5% (n patients)", str(cat_counts['<5%'])],
        ["Sidero Count 5-15% (n patients)", str(cat_counts['5-15%'])],
        ["Sidero Count > 15% (n patients)", str(cat_counts['>15%'])],
        ["Type Lame: Frottis (n slides)", str(type_counts['frottis'])],
        ["Type Lame: Ecrasement (n slides)", str(type_counts['ecrasement'])],
        ["", ""],
        ["Dataset Splits (Patients)", ""]
    ]

    for s in ['train', 'valid', 'test']:
        data.extend([
            [f"{s.capitalize()} Split", ""],
            ["    Sidero Count < 5%", str(split_stats[s]['<5%'])],
            ["    Sidero Count 5-15%", str(split_stats[s]['5-15%'])],
            ["    Sidero Count > 15%", str(split_stats[s]['>15%'])]
        ])
        
    data.extend([["", ""], ["Classification Set Sub-Images", ""]])
    for s in ['train', 'valid', 'test']:
        data.extend([
            [f"{s.capitalize()} Split", ""],
            ["    SC (n)", str(class_stats[s]['SC'])],
            ["    SN (n)", str(class_stats[s]['SN'])]
        ])

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0 or data[i][1] == "":
            cell.set_text_props(weight='bold')

    plt.savefig('clinical_table.pdf', format='pdf', bbox_inches='tight')
    print("Tableau PDF généré avec succès : clinical_table.pdf")

if __name__ == '__main__':
    main()
