#!/usr/bin/env python
# coding: utf-8

# # Reproduce Scientific Plots
# 
# This notebook reproduces the 6-panel figure using data from `FatigueData-CMA2022.xlsx`.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


# ## 1. Load and Preprocess Data

# In[ ]:


file_path = 'FatigueData-CMA2022.xlsx'

# Load mapping parameters (Material names, classes)
df_params = pd.read_excel(file_path, sheet_name='parameter', header=1)

# Rename crucial columns for easier access
df_params = df_params.rename(columns={
    'dataset id': 'id', 
    'original name of the material': 'material',
    'type of the material': 'type',
    'Young\'s modulus \n(GPa)': 'E',
    'yield strength \n(MPa)': 'YS',
    'ultimate tensile strength \n(MPa)': 'UTS',
    'elongation \n(%)': 'EL',
    'calculated density': 'density' # assuming this might exist, else check columns
})

print("Material Types available:", df_params['type'].unique())


# In[ ]:


# Load Data Sheets
df_sn = pd.read_excel(file_path, sheet_name='S-N')
df_en = pd.read_excel(file_path, sheet_name='e-N')
df_dadn = pd.read_excel(file_path, sheet_name='dadn')

# Standardize column names
df_sn.columns = ['id', 'N', 'stress', 'runout']
df_en.columns = ['id', 'N', 'strain', 'runout']
df_dadn.columns = ['id', 'dK', 'dadN']

# Merge material type into dataframes
df_sn = df_sn.merge(df_params[['id', 'material', 'type']], on='id', how='left')
df_en = df_en.merge(df_params[['id', 'material', 'type']], on='id', how='left')
df_dadn = df_dadn.merge(df_params[['id', 'material', 'type']], on='id', how='left')

print("S-N Data Types:", df_sn['type'].unique())


# ## 2. Generate Plots

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

# --- Panel A: PCA of Material Properties ---
# Extract features for PCA
features = ['E', 'YS', 'UTS', 'EL']
df_pca = df_params.dropna(subset=features + ['type', 'material'])

if not df_pca.empty:
    X = df_pca[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    df_pca['PC1'] = components[:, 0]
    df_pca['PC2'] = components[:, 1]
    
    # Plot
    types = df_pca['type'].unique()
    markers = ['o', '^', 's', 'D', 'v', '<', '>']
    for i, t in enumerate(types):
        subset = df_pca[df_pca['type'] == t]
        axes[0].scatter(subset['PC1'], subset['PC2'], label=t, alpha=0.7, edgecolors='k', s=80, marker=markers[i % len(markers)])
    
    axes[0].set_xlabel('1st Principal Component')
    axes[0].set_ylabel('2nd Principal Component')
    axes[0].legend(loc='best')
    axes[0].set_title('(a) Material Property Space')
else:
    axes[0].text(0.5, 0.5, 'Insufficient Data for PCA', ha='center', va='center')

# --- Panel B: S-N Data (Focus on Metallic Glasses if exist, else All) ---
# Check if we have specific types similar to paper (MG, MPEA)
# Let's plot MPEA data if available separate from others, or just random subsets
# If 'type' column has 'BMG' or similar, we use that.

unique_types = df_sn['type'].dropna().unique()
# Fallback: Split by type simply
target_types_b = [t for t in unique_types if 'Glass' in str(t) or 'MG' in str(t)]
target_types_c = [t for t in unique_types if t not in target_types_b]

def plot_sn(ax, df, title_prefix):
    for mat in df['material'].dropna().unique()[:10]: # Limit to 10 materials for clarity
        subset = df[df['material'] == mat]
        if len(subset) > 5:
            ax.scatter(subset['N'], subset['stress'], label=mat[:10], s=20, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('N / cycles')
    ax.set_ylabel('$\sigma_a$ / MPa')
    ax.set_title(title_prefix)

if target_types_b:
    plot_sn(axes[1], df_sn[df_sn['type'].isin(target_types_b)], '(b) S-N Curves (MG)')
else:
    # Split roughly if no explicit labels match
    plot_sn(axes[1], df_sn.iloc[:len(df_sn)//2], '(b) S-N Curves (Set 1)')

# --- Panel C: S-N Data (MPEA or others) ---
if target_types_c:
    plot_sn(axes[2], df_sn[df_sn['type'].isin(target_types_c)], '(c) S-N Curves (MPEA/Alloys)')
else:
    plot_sn(axes[2], df_sn.iloc[len(df_sn)//2:], '(c) S-N Curves (Set 2)')

# --- Panel D: Strain-Life (e-N) ---
# Plot all available e-N data
for mat in df_en['material'].dropna().unique()[:8]:
    subset = df_en[df_en['material'] == mat]
    axes[3].plot(subset['N'], subset['strain'], 'o', alpha=0.6, label=mat[:8])

axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].set_xlabel('N / cycles')
axes[3].set_ylabel('$\epsilon_a$')
axes[3].set_title('(d) Strain-Life')

# --- Panel E: da/dN vs dK ---
# Plot all available da/dN data
for mat in df_dadn['material'].dropna().unique()[:5]: # Heavy data, limit number of series
    subset = df_dadn[df_dadn['material'] == mat]
    axes[4].plot(subset['dK'], subset['dadN'], '.', ms=2, alpha=0.5, label=mat[:8])

axes[4].set_xscale('log')
axes[4].set_yscale('log')
axes[4].set_xlabel('$\Delta K$ / MPa $\sqrt{m}$')
axes[4].set_ylabel('$da/dN$ / m/cycle')
axes[4].set_title('(e) Crack Growth Rate')

# --- Panel F: Simple Correlation (Fitting) ---
# Let's fit S-N data (Power law: sigma = A * N^B) and plot parameters if possible
fit_results = []
for mid, group in df_sn.groupby('id'):
    if len(group) > 4:
        try:
            # linearized: log(sigma) = log(A) + B*log(N)
            # y = C + B*x
            log_n = np.log10(group['N'])
            log_s = np.log10(group['stress'])
            coeffs = np.polyfit(log_n, log_s, 1)
            B, log_A = coeffs
            fit_results.append({'id': mid, 'A': 10**log_A, 'B': B})
        except:
            pass

df_fit = pd.DataFrame(fit_results)
if not df_fit.empty:
    # Plot A vs B or similar
    axes[5].scatter(df_fit['B'], df_fit['A'], alpha=0.6, edgecolors='k')
    axes[5].set_xlabel('Basquin Exponent, $b$')
    axes[5].set_ylabel('Fatigue Strength Coefficient, $\sigma_f\' (A)$')
    axes[5].set_yscale('log')
    axes[5].set_title('(f) Basquin Parameters Fit')
else:
    axes[5].text(0.5, 0.5, 'Insufficient Data for Fitting', ha='center', va='center')

plt.tight_layout()
plt.show()

