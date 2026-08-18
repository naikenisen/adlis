import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

inference_csv = os.path.join(project_root, "dataset/inference-test-externe.csv")
ground_truth_csv = os.path.join(project_root, "dataset/test-externe.csv")
output_figure = os.path.join(project_root, "figures/figure_3.png")

def main():
    print("Generating Figure 3...")
    if not os.path.exists(inference_csv) or not os.path.exists(ground_truth_csv):
        print(f"Missing CSV files for plotting:\n- {inference_csv}\n- {ground_truth_csv}")
        return
        
    df_pred = pd.read_csv(inference_csv)
    if 'SC' in df_pred.columns and 'SN' in df_pred.columns:
        df_pred = df_pred[(df_pred['SC'] + df_pred['SN']) >= 100]
    df_gt = pd.read_csv(ground_truth_csv)
    
    # Safe string conversion to merge patient IDs correctly
    df_pred['id'] = df_pred['id'].astype(str).str.strip().str[:10]
    df_gt['id'] = df_gt['id'].astype(str).str.strip().str[:10]
    
    df = pd.merge(df_gt, df_pred, on='id', how='inner')
    
    if df.empty:
        print("Merged dataframe is empty! Check the IDs in both CSVs.")
        return
        
    valeur = df['valeur'].values
    prediction = df['prediction'].values
    
    # Classification: > 15% is positive
    y_true = (valeur > 15).astype(int)
    y_pred = (prediction > 15).astype(int)
    y_scores = prediction
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative (<=15%)', 'Positive (>15%)'])
    disp.plot(ax=ax1, cmap='Blues', colorbar=False)
    ax1.set_title('Confusion Matrix')
    
    # Plot 2: ROC Curve
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close()
    print(f"Figure saved to {output_figure}")

if __name__ == "__main__":
    main()
