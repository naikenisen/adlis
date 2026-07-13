import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_selection(selection_file: Path):
    paths = []
    with selection_file.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(Path(p))
    return paths


def build_board(image_paths, output_file: Path, title: str, ncols: int = 4):
    valid_paths = [p for p in image_paths if p.exists()]
    missing_paths = [p for p in image_paths if not p.exists()]

    if not valid_paths:
        raise ValueError("Aucune image valide dans la selection.")

    n = len(valid_paths)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes = np.array(axes).reshape(-1)

    for i, path in enumerate(valid_paths):
        img = plt.imread(path)
        axes[i].imshow(img)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return len(valid_paths), len(missing_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Cree un board PNG a partir d'un fichier de selection d'images."
    )
    parser.add_argument(
        "--selection",
        default="selection.txt",
        help="Fichier texte contenant un chemin d'image par ligne.",
    )
    parser.add_argument(
        "--output",
        default="selection_board.png",
        help="Nom du fichier PNG de sortie.",
    )
    parser.add_argument(
        "--title",
        default="Board depuis selection",
        help="Titre affiche sur le board.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Nombre de colonnes dans la grille.",
    )

    args = parser.parse_args()

    selection_file = Path(args.selection)
    if not selection_file.exists():
        raise FileNotFoundError(f"Fichier introuvable: {selection_file}")

    image_paths = load_selection(selection_file)
    if not image_paths:
        raise ValueError("Le fichier de selection est vide.")

    output_file = Path(args.output)
    valid_count, missing_count = build_board(
        image_paths=image_paths,
        output_file=output_file,
        title=args.title,
        ncols=args.ncols,
    )

    print(f"Board genere: {output_file}")
    print(f"Images utilisees: {valid_count}")
    if missing_count > 0:
        print(f"Images manquantes ignorees: {missing_count}")


if __name__ == "__main__":
    main()
