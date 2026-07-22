import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

inference_csv = os.path.join(project_root, "dataset/inference-test-externe.csv")
ground_truth_csv = os.path.join(project_root, "dataset/test-externe.csv")
output_figure = os.path.join(project_root, "figures/figure_2C.png")

def main():
    print("Generating plots...")
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
    
    # Bland-Altman
    mean = np.mean([valeur, prediction], axis=0)
    diff = prediction - valeur
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Linear Regression
    ax1.scatter(valeur, prediction, alpha=0.7, color='#1F77B4')
    
    reg = LinearRegression().fit(valeur.reshape(-1, 1), prediction)
    line_x = np.array([min(valeur), max(valeur)])
    line_y = reg.predict(line_x.reshape(-1, 1))
    ax1.plot(line_x, line_y, color='red', label='Regression Line')
    
    ideal_x = np.array([0, max(valeur)])
    ax1.plot(ideal_x, ideal_x, color='gray', linestyle='--', label='y = x')
    
    r, p = pearsonr(valeur, prediction)
    ax1.text(0.05, 0.95, f'r = {r:.2f}\np = {p:.2e}', transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    ax1.set_xlabel('Ground Truth (%)')
    ax1.set_ylabel('Prediction (%)')
    ax1.set_title('Linear Regression')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Bland-Altman
    ax2.scatter(mean, diff, alpha=0.7, color='#1F77B4')
    ax2.axhline(md, color='red', linestyle='-', label=f'Mean Diff = {md:.2f}')
    ax2.axhline(md + 1.96*sd, color='gray', linestyle='--', label=f'+1.96 SD = {md + 1.96*sd:.2f}')
    ax2.axhline(md - 1.96*sd, color='gray', linestyle='--', label=f'-1.96 SD = {md - 1.96*sd:.2f}')
    
    ax2.set_xlabel('Mean of GT and Prediction (%)')
    ax2.set_ylabel('Difference (Prediction - GT) (%)')
    ax2.set_title('Bland-Altman Plot')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close()
    print(f"Figure saved to {output_figure}")

if __name__ == "__main__":
    main()
