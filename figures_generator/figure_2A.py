import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# 1. Configuration des données (Extraites de vos images A, B, C)
# Structure: [[TP, FN], [FP, TN]]
# Note: Dans vos images, l'axe Y est True Label (SC en haut, SN en bas)
# et l'axe X est Predicted Label (SC à gauche, SN à droite).
# Donc:
# Case (0,0) = True SC / Pred SC (TP si on considère SC comme classe positive)
# Case (0,1) = True SC / Pred SN (FN)
# Case (1,0) = True SN / Pred SC (FP)
# Case (1,1) = True SN / Pred SN (TN)

data = {
    "Training Set": np.array([[440, 73], 
                              [134, 1909]]),
    
    "Validation Set": np.array([[123, 13], 
                                [25, 414]]),
    
    "Test Set": np.array([[18, 2], 
                          [23, 225]])
}

labels = ["SC", "SN"] # SC = Ring Sideroblasts, SN = Normal Sideroblasts

# 2. Fonction pour calculer les métriques
def calculate_metrics(cm):
    # cm[0,0] = TP, cm[0,1] = FN, cm[1,0] = FP, cm[1,1] = TN
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    
    # Calculs
    accuracy = (TP + TN) / np.sum(cm)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0 # Rappel (Recall) pour la classe SC
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0 # Rappel pour la classe SN
    
    return accuracy, sensitivity, specificity

# 3. Paramètres de style pour un rendu type « grandes revues »
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,             # 7–9 pt classique
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,   # texte vectoriel dans PDF/PS
    'ps.fonttype': 42,
    'svg.fonttype': 'none'
})

# Normalisation par ligne (proportions) pour comparabilité visuelle
def row_normalize(cm: np.ndarray) -> np.ndarray:
    row_sum = cm.sum(axis=1, keepdims=True)
    # éviter division par zéro
    row_sum[row_sum == 0] = 1
    return cm / row_sum

# Générer les annotations counts + %
def make_annotation(cm: np.ndarray) -> np.ndarray:
    pct = row_normalize(cm)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i, j])}\n({pct[i, j]*100:.0f}%)"
    return annot

# 4. Création de la figure (largeur double colonne ~180 mm)
mm = 1/25.4
fig_w, fig_h = 180*mm, 65*mm
fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), constrained_layout=True)

# 5. Tracer les heatmaps (mêmes bornes 0–1) + une colorbar partagée
mappable_for_cbar = None
for idx, (ax, (dataset_name, matrix)) in enumerate(zip(axes, data.items())):
    acc, sens, spec = calculate_metrics(matrix)
    pct = row_normalize(matrix)
    annot = make_annotation(matrix)

    sns.set_theme(context='paper', style='ticks')
    hm = sns.heatmap(
        pct,
        annot=annot,
        fmt='',
        cmap='Blues',
        vmin=0, vmax=1,
        ax=ax,
        cbar=False,  # colorbar commune ajoutée après
        square=True,
        annot_kws={"size": 7}
    )

    # Labels sobres
    ax.set_title(dataset_name, pad=6, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)

    # Légère ligne de métriques sous chaque panneau
    ax.text(
        0.5, -0.22,
        f"Acc {acc:.2f} • Sens {sens:.2f} • Spec {spec:.2f}",
        transform=ax.transAxes, ha='center', va='top', fontsize=7
    )

    # Garder un mappable pour créer une colorbar commune
    if idx == 2:
        mappable_for_cbar = ax.collections[0]

# Colorbar partagée
if mappable_for_cbar is not None:
    cbar = fig.colorbar(
        mappable_for_cbar, ax=axes, location='right', fraction=0.03, pad=0.02
    )
    cbar.set_label('Row-normalized proportion')

# 6. Lettres de panneaux A, B, C
panel_letters = ['A', 'B', 'C']
for ax, letter in zip(axes, panel_letters):
    ax.text(-0.18, 1.03, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')

# 7. Sauvegardes (PNG + vectoriels)
plt.savefig('confusion_matrices_metrics.png', bbox_inches='tight')
plt.savefig('confusion_matrices_metrics.pdf', bbox_inches='tight')
plt.savefig('confusion_matrices_metrics.svg', bbox_inches='tight')
plt.show()
