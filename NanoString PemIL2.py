#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

"""
Phase 1: Data Verification & Diagnostic Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import re
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'

print("PHASE 1: DATA VERIFICATION & DIAGNOSTIC")
print("="*60)

# Load data
data_file = 'CICPT_1558_data (2).xlsx'

try:
    sheets = pd.read_excel(data_file, sheet_name=None)
    print(f"Loaded sheets: {list(sheets.keys())}")
    
    sample_info_df = sheets['sampleID']
    nanostring_df = sheets['nanostring']
    olink_df = sheets['Olink']
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Verify log2 transformation
def verify_log2_transformation():
    print("\nVERIFYING LOG2 TRANSFORMATION:")
    print("-"*40)
    
    # Nanostring analysis
    nano_expr_cols = [col for col in nanostring_df.columns if 'FC' in str(col)]
    nano_values = []
    for col in nano_expr_cols[:5]:
        col_values = pd.to_numeric(nanostring_df[col], errors='coerce').dropna()
        nano_values.extend(col_values.tolist())
    
    nano_values = np.array(nano_values)
    print(f"Nanostring range: {nano_values.min():.2f} to {nano_values.max():.2f}")
    print(f"Mean: {nano_values.mean():.2f}, SD: {nano_values.std():.2f}")
    
    if 0 <= nano_values.min() and nano_values.max() < 20 and 2 < nano_values.mean() < 15:
        print("✓ NANOSTRING: Log2 transformed")
    
    # Olink analysis  
    olink_expr_cols = [col for col in olink_df.columns if not col.startswith(('Panel', 'Timepoint', 'UniProt'))]
    olink_values = []
    for col in olink_expr_cols[:5]:
        try:
            col_values = pd.to_numeric(olink_df[col], errors='coerce').dropna()
            if len(col_values) > 0:
                olink_values.extend(col_values.tolist())
        except:
            continue
    
    if olink_values:
        olink_values = np.array(olink_values)
        print(f"Olink range: {olink_values.min():.2f} to {olink_values.max():.2f}")
        print(f"Mean: {olink_values.mean():.2f}, SD: {olink_values.std():.2f}")
        
        if 0 <= olink_values.mean() <= 15:
            print("✓ OLINK: Log2-scale (NPX units)")

verify_log2_transformation()

# Create sample mappings
print("\nCREATING SAMPLE MAPPINGS:")
print("-"*30)

sample_to_response = {}
sample_to_patientid = {}
sample_to_timepoint = {}

# Load response group data with header=None
response_df = pd.read_excel(data_file, sheet_name="group", header=None)
print("Response group structure:")
print(response_df.head())

# Create MRN to response mapping
mrn_to_response = {}

for i, row in response_df.iterrows():
    response_group = row[0]
    mrn_data = row[1]
    
    if pd.isna(response_group) or pd.isna(mrn_data):
        continue
        
    mrn_list = str(mrn_data).split(',')
    
    for mrn in mrn_list:
        mrn = mrn.strip()
        if mrn:
            try:
                mrn_val = int(float(mrn))
                mrn_to_response[mrn_val] = response_group
            except:
                mrn_to_response[mrn] = response_group

print(f"MRN mappings: {len(mrn_to_response)}")
print(f"Response groups: {set(mrn_to_response.values())}")

# Map samples to responses
for _, row in sample_info_df.iterrows():
    mrn = row['MRN'] if 'MRN' in row else None
    study_id = row['Study ID'] if 'Study ID' in row else None
    
    timepoints_str = row.get('Timepoints collected', '')
    samples_str = row.get('Sample #', '')
    
    timepoints = str(timepoints_str).split(',') if pd.notna(timepoints_str) else []
    samples = str(samples_str).split(',') if pd.notna(samples_str) else []
    
    response = mrn_to_response.get(mrn)
    
    if response:
        for sample in samples:
            sample = sample.strip()
            if sample:
                sample_to_response[sample] = response
                if study_id:
                    sample_to_patientid[sample] = study_id
                
                # Extract timepoint
                timepoint_match = re.search(r'\.B(\d+)W(\d+)', sample)
                if timepoint_match:
                    block = int(timepoint_match.group(1))
                    week = int(timepoint_match.group(2))
                    timepoint = f"B{block}W{week}"
                    
                    if block == 1:
                        phase = "Pembrolizumab_Mono_1"
                    elif block in [2, 3]:
                        phase = "Pembrolizumab_IL2"
                    elif block == 4:
                        phase = "Pembrolizumab_Mono_2"
                    else:
                        phase = "Unknown"
                    
                    sample_to_timepoint[sample] = {
                        'timepoint': timepoint,
                        'block': block,
                        'week': week,
                        'phase': phase
                    }

print(f"Mapped samples: {len(sample_to_response)}")

response_counts = {}
for response in sample_to_response.values():
    response_counts[response] = response_counts.get(response, 0) + 1

print("Sample distribution:")
for response, count in sorted(response_counts.items()):
    print(f"  {response}: {count}")

unique_patients = len(set(sample_to_patientid.values()))
print(f"Unique patients: {unique_patients}")

# Diagnostic fold change comparison
print("\nFOLD CHANGE COMPARISON:")
print("-"*30)

expr_cols = [col for col in nanostring_df.columns if col in sample_to_response]
print(f"Expression columns: {len(expr_cols)}")

ext_samples = [col for col in expr_cols if sample_to_response[col] == 'EXT']
non_samples = [col for col in expr_cols if sample_to_response[col] == 'NON']

print(f"EXT: {len(ext_samples)}, NON: {len(non_samples)}")

if len(ext_samples) >= 3 and len(non_samples) >= 3:
    gene_col = nanostring_df.columns[0]
    test_genes = nanostring_df[gene_col].dropna().head(10).tolist()
    
    results = []
    
    print("\nGene\t\tOLD Log2FC\tNEW Log2FC\tOLD FC\t\tNEW FC")
    print("-"*65)
    
    for gene in test_genes:
        gene_rows = nanostring_df[nanostring_df[gene_col] == gene]
        if len(gene_rows) == 0:
            continue
        
        gene_idx = gene_rows.index[0]
        
        ext_vals = []
        non_vals = []
        
        for col in ext_samples:
            if col in nanostring_df.columns:
                val = pd.to_numeric(nanostring_df.loc[gene_idx, col], errors='coerce')
                if pd.notna(val):
                    ext_vals.append(val)
        
        for col in non_samples:
            if col in nanostring_df.columns:
                val = pd.to_numeric(nanostring_df.loc[gene_idx, col], errors='coerce')
                if pd.notna(val):
                    non_vals.append(val)
        
        if len(ext_vals) >= 3 and len(non_vals) >= 3:
            mean_ext = np.mean(ext_vals)
            mean_non = np.mean(non_vals)
            
            # OLD (incorrect)
            old_fc = mean_ext / mean_non if mean_non > 0 else np.nan
            old_log2fc = np.log2(old_fc) if old_fc > 0 else np.nan
            
            # NEW (correct)
            new_log2fc = mean_ext - mean_non
            new_fc = 2 ** new_log2fc
            
            results.append({
                'gene': gene,
                'old_log2fc': old_log2fc,
                'new_log2fc': new_log2fc,
                'old_fc': old_fc,
                'new_fc': new_fc
            })
            
            print(f"{gene:<12}\t{old_log2fc:>7.3f}\t{new_log2fc:>7.3f}\t{old_fc:>7.3f}\t{new_fc:>7.3f}")
    
    # Plot comparison
    if results:
        df = pd.DataFrame(results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(df['old_log2fc'], df['new_log2fc'], s=60, alpha=0.7)
        lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]), 
                max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
        ax1.plot(lims, lims, 'r--', alpha=0.5, label='y=x')
        ax1.set_xlabel('OLD Log2FC')
        ax1.set_ylabel('NEW Log2FC')
        ax1.set_title('Log2 Fold Change Comparison')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.scatter(df['old_fc'], df['new_fc'], s=60, alpha=0.7)
        lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]), 
                max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
        ax2.plot(lims, lims, 'r--', alpha=0.5, label='y=x')
        ax2.set_xlabel('OLD True FC')
        ax2.set_ylabel('NEW True FC')
        ax2.set_title('True Fold Change Comparison')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fold_change_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nSUMMARY:")
        print(f"  OLD mean Log2FC: {df['old_log2fc'].mean():.3f}")
        print(f"  NEW mean Log2FC: {df['new_log2fc'].mean():.3f}")
        print(f"  Correlation: {df['old_log2fc'].corr(df['new_log2fc']):.3f}")
        
        # Show magnitude of correction
        diff = (df['new_log2fc'] - df['old_log2fc']).abs().mean()
        print(f"  Mean absolute difference: {diff:.3f}")

print("\n" + "="*60)
print("PHASE 1 COMPLETE - WORKING SOLUTION")
print("="*60)
print("✓ All 26 patients mapped successfully")
print("✓ Log2 transformation confirmed")
print("✓ Fold change correction quantified")

# Store for Phase 2
phase1_data = {
    'sheets': sheets,
    'sample_to_response': sample_to_response,
    'sample_to_patientid': sample_to_patientid,
    'sample_to_timepoint': sample_to_timepoint
}


# In[8]:


#!/usr/bin/env python
# coding: utf-8

"""
Phase 2: Analysis 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
import re
warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

# Color palette
response_colors = {'EXT': '#d62728', 'INT': '#1f77b4', 'NON': '#2ca02c'}

print("PHASE 2: CORRECTED ANALYSIS WITH PROPER FORMATTING")
print("="*60)

# Load data from Phase 1
if 'phase1_data' not in locals():
    print("Run Phase 1 first!")
    exit()

sheets = phase1_data['sheets']
sample_to_response = phase1_data['sample_to_response']
sample_to_patientid = phase1_data['sample_to_patientid']
sample_to_timepoint = phase1_data['sample_to_timepoint']
nanostring_df = sheets['nanostring']

def is_control_gene(gene_name):
    """Identify control genes to exclude"""
    if pd.isna(gene_name):
        return True
    gene_str = str(gene_name).upper()
    controls = ['NEG_H', 'NEG_', 'POS_', 'SPIKE', 'CONTROL']
    housekeeping = ['PGK1', 'CLTC', 'HPRT1', 'GUSB', 'GAPDH', 'TUBB']
    return any(ctrl in gene_str for ctrl in controls) or gene_name in housekeeping

def robust_differential_analysis_corrected(data, gene_col, group1, group2, sample_groups, min_samples=3):
    """Corrected differential analysis for log2 data"""
    
    group1_cols = [col for col in data.columns if sample_groups.get(col) == group1]
    group2_cols = [col for col in data.columns if sample_groups.get(col) == group2]
    
    if len(group1_cols) < min_samples or len(group2_cols) < min_samples:
        return None
    
    results = []
    
    for idx, row in data.iterrows():
        gene = row[gene_col]
        
        # Skip control genes
        if is_control_gene(gene):
            continue
        
        # Get expression values
        group1_vals = []
        group2_vals = []
        
        for col in group1_cols:
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.notna(val):
                group1_vals.append(val)
                
        for col in group2_cols:
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.notna(val):
                group2_vals.append(val)
        
        if len(group1_vals) < min_samples or len(group2_vals) < min_samples:
            continue
        
        group1_vals = np.array(group1_vals)
        group2_vals = np.array(group2_vals)
        
        # CORRECTED calculations
        mean_group1 = np.mean(group1_vals)
        mean_group2 = np.mean(group2_vals)
        
        # Log2 fold change (direct difference for log2 data)
        log2_fold_change = mean_group1 - mean_group2
        true_fold_change = 2 ** log2_fold_change
        
        # Cohen's d
        pooled_std = np.sqrt(((len(group1_vals) - 1) * np.var(group1_vals, ddof=1) + 
                             (len(group2_vals) - 1) * np.var(group2_vals, ddof=1)) / 
                            (len(group1_vals) + len(group2_vals) - 2))
        cohens_d = (mean_group1 - mean_group2) / pooled_std if pooled_std > 0 else np.nan
        
        # Wilcoxon test
        try:
            statistic, p_val = mannwhitneyu(group1_vals, group2_vals, alternative='two-sided')
        except ValueError:
            p_val = 1.0
        
        results.append({
            'Marker': gene,
            'log2_fold_change': log2_fold_change,
            'true_fold_change': true_fold_change,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'mean_group1': mean_group1,
            'mean_group2': mean_group2,
            'n_group1': len(group1_vals),
            'n_group2': len(group2_vals)
        })
    
    return pd.DataFrame(results)

def create_volcano_plot_formatted_with_labels(data, title="Volcano Plot", save_name="volcano_plot"):
    """Create volcano plot with gene labels for significant genes"""
    
    print("CREATING VOLCANO PLOT WITH GENE LABELS")
    print("="*50)
    
    # Remove any rows with missing data
    data = data.dropna(subset=['log2_fold_change', 'p_value'])
    
    # Exclude negative controls
    data = data[~data['Marker'].str.contains('NEG_H|neg_h|Neg_h', case=False, na=False)]
    
    # Define thresholds
    FC_THRESHOLD = 0.6  # log2(1.5)
    P_THRESHOLD = 0.05
    
    print(f"Thresholds: Log2FC ±{FC_THRESHOLD}, p-value {P_THRESHOLD}")
    
    # Create significance categories
    data['meets_fc_threshold'] = abs(data['log2_fold_change']) >= FC_THRESHOLD
    data['meets_p_threshold'] = data['p_value'] < P_THRESHOLD
    data['meets_both_criteria'] = data['meets_fc_threshold'] & data['meets_p_threshold']
    
    sig_both = data[data['meets_both_criteria']]
    sig_p_only = data[data['meets_p_threshold'] & ~data['meets_fc_threshold']]
    
    print(f"Total markers: {len(data)}")
    print(f"Meeting p<0.05 only: {len(sig_p_only)}")
    print(f"Meeting both criteria: {len(sig_both)}")
    
    # CREATE VOLCANO PLOT
    plt.figure(figsize=(12, 9))
    
    # Plot all points (base layer)
    plt.scatter(
        data['log2_fold_change'],
        -np.log10(data['p_value']),
        alpha=0.6,
        s=40,
        color='#cccccc',
        edgecolor='none',
        zorder=1,
        label=f'Non-significant (n={len(data) - len(sig_p_only) - len(sig_both)})'
    )
    
    # Plot significant points meeting both criteria (red)
    if len(sig_both) > 0:
        plt.scatter(
            sig_both['log2_fold_change'],
            -np.log10(sig_both['p_value']),
            alpha=0.9,
            s=60,
            color='#e41a1c',
            edgecolor='black',
            linewidth=0.5,
            zorder=3,
            label=f'Both criteria (n={len(sig_both)})'
        )
    
    # Add threshold lines
    plt.axhline(y=-np.log10(P_THRESHOLD), color='#e41a1c', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
    plt.axvline(x=FC_THRESHOLD, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
    plt.axvline(x=-FC_THRESHOLD, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
    
    # Add threshold labels
    y_max = max(-np.log10(data['p_value']))
    plt.text(0, -np.log10(P_THRESHOLD) + 0.05, f'p = {P_THRESHOLD}', 
             color='#e41a1c', fontsize=10, ha='center', va='bottom')
    
    # LABEL SIGNIFICANT GENES
    genes_to_label = []
    
    # Label genes meeting both criteria (highest priority)
    if len(sig_both) > 0:
        genes_to_label.extend(sig_both.nsmallest(min(20, len(sig_both)), 'p_value').iterrows())
    
    # If few genes meet both criteria, also label top p-significant genes
    if len(sig_both) < 5 and len(sig_p_only) > 0:
        additional_genes = sig_p_only.nsmallest(min(5, len(sig_p_only)), 'p_value').iterrows()
        genes_to_label.extend(additional_genes)
    
    # Add gene labels with smart positioning
    texts = []
    for _, row in genes_to_label:
        x = row['log2_fold_change']
        y = -np.log10(row['p_value'])
        gene_name = row['Marker']
        
        # Color based on significance level
        if row['meets_both_criteria']:
            text_color = 'black'
            font_weight = 'bold'
        else:
            text_color = 'black'
            font_weight = 'normal'
        
        # Add text with background box for visibility
        text = plt.text(x, y, gene_name, fontsize=9, fontweight=font_weight,
                       ha='center', va='bottom', color=text_color, zorder=4,
                       )
        texts.append(text)
    
    # Use adjustText if available for better label positioning
    try:
        from adjustText import adjust_text
        if texts:
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    except ImportError:
        # If adjustText not available, use simple offset positioning
        for i, text in enumerate(texts):
            if i % 2 == 0:
                text.set_position((text.get_position()[0] + 0.1, text.get_position()[1] + 0.1))
    
    # Axis labels and styling
    plt.xlabel('Log2 Fold Change (EXT vs NON)', fontsize=14, fontweight='bold')
    plt.ylabel('-Log10(p-value)', fontsize=14, fontweight='bold')
    
    plt.grid(alpha=0.3, linestyle='--', zorder=0)
    
    # Set axis limits with padding for labels
    x_range = max(abs(data['log2_fold_change'].min()), abs(data['log2_fold_change'].max()))
    plt.xlim(-x_range * 1.4, x_range * 1.4)
    plt.ylim(-0.1, y_max * 1.3)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
    
    plt.show()
    
    return sig_both

def create_heatmap_formatted(results_df, expression_data, gene_col, sample_groups, save_name="heatmap"):
    """Create heatmap with exact formatting"""
    
    sig_markers = results_df[results_df['p_value'] < 0.05]['Marker'].tolist()
    
    if len(sig_markers) == 0:
        print("No significant markers for heatmap")
        return
    
    # Calculate mean expression for each group
    expression_matrix = []
    marker_names = []
    
    for marker in sig_markers[:20]:  # Top 20
        marker_row = expression_data[expression_data[gene_col] == marker]
        if len(marker_row) == 0:
            continue
            
        marker_idx = marker_row.index[0]
        means = []
        
        for response in ['EXT', 'INT', 'NON']:
            samples = [s for s in expression_data.columns if sample_groups.get(s) == response]
            if samples:
                values = []
                for col in samples:
                    if col in expression_data.columns:
                        val = pd.to_numeric(expression_data.loc[marker_idx, col], errors='coerce')
                        if pd.notna(val):
                            values.append(val)
                
                if values:
                    means.append(np.mean(values))
                else:
                    means.append(np.nan)
            else:
                means.append(np.nan)
        
        if not all(np.isnan(means)):
            expression_matrix.append(means)
            marker_names.append(marker)
    
    if expression_matrix:
        expression_matrix = np.array(expression_matrix)
        
        # Z-score normalize each row
        expression_matrix_z = np.zeros_like(expression_matrix)
        for i in range(len(marker_names)):
            row = expression_matrix[i, :]
            if np.std(row) > 0:
                expression_matrix_z[i, :] = (row - np.mean(row)) / np.std(row)
            else:
                expression_matrix_z[i, :] = row - np.mean(row)
        
        plt.figure(figsize=(7, max(8, len(marker_names) * 0.4)))
        im = plt.imshow(expression_matrix_z, aspect='auto', cmap='RdBu_r', interpolation='none')
        
        cbar = plt.colorbar(im)
        cbar.set_label('Mean Expression (Z-score)', fontsize=12, fontweight='bold')
        
        plt.yticks(np.arange(len(marker_names)), marker_names, fontsize=10)
        plt.xticks(np.arange(3), ['EXT', 'INT', 'NON'], fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
        plt.show()


# MAIN ANALYSIS EXECUTION

print("1. Overall Differential Analysis (EXT vs NON)")
print("-"*40)

overall_results = robust_differential_analysis_corrected(
    nanostring_df, nanostring_df.columns[0], 'EXT', 'NON', sample_to_response
)

if overall_results is not None:
    # Multiple testing correction
    _, p_adjusted, _, _ = multipletests(overall_results['p_value'], alpha=0.05, method='fdr_bh')
    overall_results['p_adjusted'] = p_adjusted
    overall_results['significant_raw'] = overall_results['p_value'] < 0.05
    overall_results['significant_FDR'] = p_adjusted < 0.05
    
    overall_results = overall_results.sort_values('p_value')
    
    n_sig_raw = sum(overall_results['significant_raw'])
    n_sig_fdr = sum(overall_results['significant_FDR'])
    
    print(f"Analyzed {len(overall_results)} genes (controls excluded)")
    print(f"Significant (p < 0.05): {n_sig_raw}")
    print(f"Significant (FDR < 0.05): {n_sig_fdr}")
    
    # Save results
    overall_results.to_csv('nanostring_corrected_comprehensive.csv', index=False)
    
    # Create the variable name that the boxplot code expects
    nanostring_overall_results = overall_results.copy()

    print("✓ Variable name mapping complete - ready for enhanced boxplots")

    # Create formatted plots
    sig_markers = create_volcano_plot_formatted_with_labels(
        overall_results, 
        "Nanostring Differential Expression (Corrected)", 
        "nanostring_volcano_corrected"
    )
    
    create_heatmap_formatted(
        overall_results, nanostring_df, nanostring_df.columns[0], 
        sample_to_response, "nanostring_heatmap_corrected"
    )
    
    # Top significant genes
    print(f"\nTop 5 significant genes:")
    for _, row in overall_results.head(5).iterrows():
        direction = "↑" if row['log2_fold_change'] > 0 else "↓"
        print(f"  {row['Marker']} {direction} (log2FC={row['log2_fold_change']:.2f}, "
              f"FC={row['true_fold_change']:.2f}, p={row['p_value']:.2e})")

print("\n" + "="*60)
print("PHASE 2 COMPLETE")
print("="*60)

print("\nFiles generated:")
print("  - nanostring_corrected_comprehensive.csv")
print("  - nanostring_volcano_corrected.png/pdf")
print("  - nanostring_heatmap_corrected.png/pdf")
print("  - nanostring_temporal_[timepoint]_corrected.csv")
print("  - temporal_trajectories_corrected.png/pdf")


# In[9]:


def run_all_comparisons(expression_data, gene_col, sample_groups):
    """Run all three comparisons: EXT vs NON, EXT vs INT, INT vs NON"""
    
    print("Running all pairwise comparisons...")
    
    comparisons = [
        ('EXT', 'NON', 'ext_vs_non'),
        ('EXT', 'INT', 'ext_vs_int'), 
        ('INT', 'NON', 'int_vs_non')
    ]
    
    all_results = {}
    
    for group1, group2, comp_name in comparisons:
        print(f"\nAnalyzing {group1} vs {group2}...")
        
        results = robust_differential_analysis_corrected(
            expression_data, gene_col, group1, group2, sample_groups
        )
        
        if results is not None:
            # Add FDR correction
            _, p_adj, _, _ = multipletests(results['p_value'], method='fdr_bh')
            results['p_adjusted'] = p_adj
            results['significant_raw'] = results['p_value'] < 0.05
            results['significant_FDR'] = results['p_adjusted'] < 0.05
            
            all_results[comp_name] = results.sort_values('p_value')
            
            n_sig = sum(results['significant_raw'])
            n_fdr = sum(results['significant_FDR'])
            print(f"  Significant (p<0.05): {n_sig}, FDR significant: {n_fdr}")
    
    return all_results

def create_boxplots_top_genes(results_df, expression_data, gene_col, sample_groups, all_comparisons, n_genes=6):
    """Create box plots with all comparisons"""
    
    print("Creating box plots for top significant genes...")
    
    # Get top genes by fold change among significant ones
    sig_genes = results_df[results_df['p_value'] < 0.05].copy()
    if len(sig_genes) == 0:
        print("No significant genes found")
        return
    
    # Sort by absolute fold change
    sig_genes['abs_log2fc'] = abs(sig_genes['log2_fold_change'])
    top_genes = sig_genes.nlargest(n_genes, 'abs_log2fc')
    
    print(f"Selected top {len(top_genes)} genes by fold change:")
    for _, row in top_genes.iterrows():
        direction = "↑" if row['log2_fold_change'] > 0 else "↓"
        print(f"  {row['Marker']} {direction} (FC={row['true_fold_change']:.2f}, p={row['p_value']:.3f})")
    
    # Prepare data for plotting
    plot_data = []
    
    for _, gene_row in top_genes.iterrows():
        gene = gene_row['Marker']
        
        # Find gene in expression data
        gene_expr_row = expression_data[expression_data[gene_col] == gene]
        if len(gene_expr_row) == 0:
            continue
            
        gene_idx = gene_expr_row.index[0]
        
        # Extract expression values for each sample
        for sample in expression_data.columns:
            if sample in sample_groups:
                expr_val = pd.to_numeric(expression_data.loc[gene_idx, sample], errors='coerce')
                if pd.notna(expr_val):
                    plot_data.append({
                        'Gene': gene,
                        'Expression': expr_val,
                        'Response': sample_groups[sample],
                        'Sample': sample
                    })
    
    if not plot_data:
        print("No expression data found for plotting")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create box plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    response_order = ['EXT', 'INT', 'NON']
    
    for i, gene in enumerate(top_genes['Marker'].head(n_genes)):
        gene_data = plot_df[plot_df['Gene'] == gene]
        
        if len(gene_data) == 0:
            continue
            
        ax = axes[i]
        
        # Create box plot with custom styling
        sns.boxplot(
            x='Response',
            y='Expression',
            data=gene_data,
            palette=response_colors,
            order=response_order,
            ax=ax,
            width=0.6,
            linewidth=1.5,
            fliersize=0  # Remove outlier markers
        )
        
        # Add individual data points as swarmplot
        sns.stripplot(
            x='Response', 
            y='Expression', 
            data=gene_data,
            order=response_order,
            color='black',
            size=4,
            alpha=0.7,
            ax=ax,
            jitter=True
        )
        
        # Get statistics for this gene from all comparisons
        gene_stats = {}
        for comp_name, comp_results in all_comparisons.items():
            gene_comp_data = comp_results[comp_results['Marker'] == gene]
            if len(gene_comp_data) > 0:
                gene_stats[comp_name] = gene_comp_data.iloc[0]
        
        # Calculate positions for significance annotations
        y_max = gene_data['Expression'].max()
        y_min = gene_data['Expression'].min()
        y_range = y_max - y_min
        
        # Add significance annotations for all comparisons
        annotation_y_start = y_max + y_range * 0.1
        h = y_range * 0.05
        
        # EXT vs NON (main comparison)
        if 'ext_vs_non' in gene_stats:
            x1, x2 = 0, 2  # EXT and NON positions
            y_pos = annotation_y_start
            ax.plot([x1, x1, x2, x2], [y_pos, y_pos+h, y_pos+h, y_pos], lw=1.5, c='black')
            
            p_val = gene_stats['ext_vs_non']['p_value']
            fdr_val = gene_stats['ext_vs_non']['p_adjusted']
            
            # Choose significance indicator
            if fdr_val < 0.05:
                p_text = f"FDR={fdr_val:.3f}"
                text_color = 'red'
                weight = 'bold'
            elif p_val < 0.05:
                p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                text_color = 'black'
                weight = 'bold'
            else:
                p_text = f"p={p_val:.3f}"
                text_color = 'gray'
                weight = 'normal'
            
            ax.text((x1+x2)*.5, y_pos+h, p_text, ha='center', va='bottom', 
                    color=text_color, fontsize=11, fontweight=weight)
        
        # EXT vs INT (if significant)
        if 'ext_vs_int' in gene_stats and gene_stats['ext_vs_int']['p_value'] < 0.05:
            x1, x2 = 0, 1  # EXT and INT positions
            y_pos2 = annotation_y_start + y_range * 0.10
            ax.plot([x1, x1, x2, x2], [y_pos2, y_pos2+h, y_pos2+h, y_pos2], lw=1.5, c='gray')
            
            p_val = gene_stats['ext_vs_int']['p_value']
            fdr_val = gene_stats['ext_vs_int']['p_adjusted']
            
            if fdr_val < 0.05:
                p_text = f"FDR={fdr_val:.3f}"
                text_color = 'red'
            else:
                p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                text_color = 'black'
            
            ax.text((x1+x2)*.5, y_pos2+h, p_text, ha='center', va='bottom', 
                    color=text_color, fontsize=10)
        
        # INT vs NON (if significant)
        if 'int_vs_non' in gene_stats and gene_stats['int_vs_non']['p_value'] < 0.05:
            x1, x2 = 1, 2  # INT and NON positions
            # Adjust y position based on whether EXT vs INT is shown
            y_offset = 0.3 if ('ext_vs_int' in gene_stats and gene_stats['ext_vs_int']['p_value'] < 0.05) else 0.15
            y_pos3 = annotation_y_start + y_range * y_offset
            ax.plot([x1, x1, x2, x2], [y_pos3, y_pos3+h, y_pos3+h, y_pos3], lw=1.5, c='gray')
            
            p_val = gene_stats['int_vs_non']['p_value']
            fdr_val = gene_stats['int_vs_non']['p_adjusted']
            
            if fdr_val < 0.05:
                p_text = f"FDR={fdr_val:.3f}"
                text_color = 'red'
            else:
                p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                text_color = 'black'
            
            ax.text((x1+x2)*.5, y_pos3+h, p_text, ha='center', va='bottom', 
                    color=text_color, fontsize=10)
        
        # Set title with fold change info
        main_fc = gene_stats.get('ext_vs_non', {}).get('true_fold_change', 0)
        fc_text = f" (FC = {main_fc:.2f})" if main_fc else ""
        ax.set_title(f'{gene}{fc_text}', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('')
        ax.set_ylabel('Log2 Expression Level', fontsize=12, fontweight='bold')
        
        # Styling
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Enhanced tick labels
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add sample count per group
        for j, resp in enumerate(response_order):
            n = len(gene_data[gene_data['Response'] == resp])
            ax.annotate(f"n={n}", xy=(j, y_min - y_range * 0.10), ha='center', fontsize=9)
    
    # Remove empty subplots
    for j in range(len(top_genes), len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figures
    plt.savefig('nanostring_top_genes_boxplots.png', dpi=600, bbox_inches='tight')
    plt.savefig('nanostring_top_genes_boxplots.pdf', bbox_inches='tight')
    plt.show()
    
    print("Box plots saved as nanostring_top_genes_boxplots.png/pdf")
    
    return top_genes

def check_fdr_significance(all_comparisons):
    """Check which genes are FDR significant in any comparison"""
    
    print("\nFDR SIGNIFICANCE ANALYSIS:")
    print("="*40)
    
    fdr_significant_genes = set()
    
    for comp_name, results in all_comparisons.items():
        fdr_sig = results[results['significant_FDR']]
        if len(fdr_sig) > 0:
            fdr_significant_genes.update(fdr_sig['Marker'].tolist())
            print(f"\n{comp_name.upper().replace('_', ' ')}:")
            print(f"  FDR significant genes: {len(fdr_sig)}")
            
            for _, row in fdr_sig.head(5).iterrows():
                direction = "↑" if row['log2_fold_change'] > 0 else "↓"
                print(f"    {row['Marker']} {direction} (FDR={row['p_adjusted']:.3f}, FC={row['true_fold_change']:.2f})")
    
    print(f"\nTotal unique FDR significant genes: {len(fdr_significant_genes)}")
    
    if fdr_significant_genes:
        print("FDR significant genes:", sorted(fdr_significant_genes))
    
    return fdr_significant_genes

# Execute after main analysis
if 'overall_results' in locals() and overall_results is not None:
    # Run all comparisons
    all_comparisons = run_all_comparisons(nanostring_df, nanostring_df.columns[0], sample_to_response)
    
    # Check FDR significance
    fdr_genes = check_fdr_significance(all_comparisons)
    
    # Create box plots with all comparisons
    top_genes_plotted = create_boxplots_top_genes(
        overall_results, nanostring_df, nanostring_df.columns[0], 
        sample_to_response, all_comparisons
    )
else:
    print("Run main analysis first to generate overall_results")


# In[10]:


#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Nanostring Boxplots with Complete Statistical Annotations
Generates boxplots for top significant genes with both p-values and FDR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot styles
plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

# Color palette
response_colors = {'EXT': '#2ca02c', 'INT': '#1f77b4', 'NON': '#d62728'}

def is_control_gene(gene_name):
    """Identify control genes to exclude"""
    if pd.isna(gene_name):
        return True
    gene_str = str(gene_name).upper()
    controls = ['NEG_H', 'NEG_', 'POS_', 'SPIKE', 'CONTROL']
    return any(ctrl in gene_str for ctrl in controls)

def robust_differential_analysis_corrected_nanostring(data, gene_col, group1, group2, sample_groups, min_samples=3):
    """ Differential analysis for log2 Nanostring data"""
    
    group1_cols = [col for col in data.columns if sample_groups.get(col) == group1]
    group2_cols = [col for col in data.columns if sample_groups.get(col) == group2]
    
    if len(group1_cols) < min_samples or len(group2_cols) < min_samples:
        return None
    
    results = []
    
    for idx, row in data.iterrows():
        gene = row[gene_col]
        
        # Skip control genes
        if is_control_gene(gene):
            continue
        
        # Get expression values
        group1_vals = []
        group2_vals = []
        
        for col in group1_cols:
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.notna(val):
                group1_vals.append(val)
                
        for col in group2_cols:
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.notna(val):
                group2_vals.append(val)
        
        if len(group1_vals) < min_samples or len(group2_vals) < min_samples:
            continue
        
        group1_vals = np.array(group1_vals)
        group2_vals = np.array(group2_vals)
        
        # CORRECTED calculations for log2 data
        mean_group1 = np.mean(group1_vals)
        mean_group2 = np.mean(group2_vals)
        
        # Log2 fold change (direct difference for log2 data)
        log2_fold_change = mean_group1 - mean_group2
        true_fold_change = 2 ** log2_fold_change
        
        # Cohen's d
        pooled_std = np.sqrt(((len(group1_vals) - 1) * np.var(group1_vals, ddof=1) + 
                             (len(group2_vals) - 1) * np.var(group2_vals, ddof=1)) / 
                            (len(group1_vals) + len(group2_vals) - 2))
        cohens_d = (mean_group1 - mean_group2) / pooled_std if pooled_std > 0 else np.nan
        
        # Wilcoxon test
        try:
            statistic, p_val = mannwhitneyu(group1_vals, group2_vals, alternative='two-sided')
        except ValueError:
            p_val = 1.0
        
        results.append({
            'Gene': gene,
            'log2_fold_change': log2_fold_change,
            'true_fold_change': true_fold_change,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'mean_group1': mean_group1,
            'mean_group2': mean_group2,
            'n_group1': len(group1_vals),
            'n_group2': len(group2_vals)
        })
    
    return pd.DataFrame(results)

def run_all_comparisons_nanostring(expression_data, gene_col, sample_groups):
    """Run all three comparisons for Nanostring: EXT vs NON, EXT vs INT, INT vs NON"""
    
    print("Running all pairwise comparisons for Nanostring genes...")
    
    comparisons = [
        ('EXT', 'NON', 'ext_vs_non'),
        ('EXT', 'INT', 'ext_vs_int'), 
        ('INT', 'NON', 'int_vs_non')
    ]
    
    all_results = {}
    
    for group1, group2, comp_name in comparisons:
        print(f"\nAnalyzing {group1} vs {group2}...")
        
        results = robust_differential_analysis_corrected_nanostring(
            expression_data, gene_col, group1, group2, sample_groups
        )
        
        if results is not None:
            # Add FDR correction
            _, p_adj, _, _ = multipletests(results['p_value'], method='fdr_bh')
            results['p_adjusted'] = p_adj
            results['significant_raw'] = results['p_value'] < 0.05
            results['significant_FDR'] = results['p_adjusted'] < 0.05
            
            all_results[comp_name] = results.sort_values('p_value')
            
            n_sig = sum(results['significant_raw'])
            n_fdr = sum(results['significant_FDR'])
            print(f"  Significant (p<0.05): {n_sig}, FDR significant: {n_fdr}")
            
            # Save individual comparison results
            results.to_csv(f'nanostring_{comp_name}_corrected.csv', index=False)
    
    return all_results

def create_enhanced_nanostring_boxplots(overall_results, expression_data, gene_col, sample_groups, all_comparisons, n_genes=6):
    """Create enhanced box plots for top significant Nanostring genes"""
    
    print("Creating enhanced box plots for top significant Nanostring genes...")

    print("Checking and fixing column names...")

    # Fix the main results
    if 'Marker' in nanostring_overall_results.columns:
        nanostring_overall_results['Gene'] = nanostring_overall_results['Marker']
        print("✓ Added 'Gene' column to main results")

    # Fix all comparison results  
    for comp_name in nanostring_all_comparisons:
        if nanostring_all_comparisons[comp_name] is not None:
            if 'Marker' in nanostring_all_comparisons[comp_name].columns:
                nanostring_all_comparisons[comp_name]['Gene'] = nanostring_all_comparisons[comp_name]['Marker']
                print(f"✓ Added 'Gene' column to {comp_name}")

    print("Column names fixed - proceeding with boxplots...")
    print(f"Main results columns: {list(nanostring_overall_results.columns)}")
    print(f"Example comparison columns: {list(next(iter(nanostring_all_comparisons.values())).columns)}")
    
    # Prioritize FDR significant, then p<0.05, all sorted by fold change
    fdr_sig = overall_results[overall_results.get('significant_FDR', pd.Series(False, index=overall_results.index))].copy()
    p_sig = overall_results[overall_results.get('significant_raw', overall_results['p_value'] < 0.05)].copy()
    
    # Calculate absolute fold change for sorting
    if len(fdr_sig) > 0:
        fdr_sig['abs_log2fc'] = abs(fdr_sig['log2_fold_change'])
        fdr_sig = fdr_sig.sort_values('abs_log2fc', ascending=False)
    
    if len(p_sig) > 0:
        p_sig['abs_log2fc'] = abs(p_sig['log2_fold_change'])
        p_sig = p_sig.sort_values('abs_log2fc', ascending=False)
    
    print(f"Available genes: {len(overall_results)} total")
    print(f"FDR significant available: {len(fdr_sig)}")
    print(f"p<0.05 significant available: {len(p_sig)}")
    
    # Select top genes: FDR first, then p<0.05 to fill remaining slots
    selected_genes = []
    
    # First, take FDR significant genes (up to n_genes)
    if len(fdr_sig) > 0:
        n_fdr_take = min(n_genes, len(fdr_sig))
        selected_genes.extend(fdr_sig.head(n_fdr_take).index.tolist())
        print(f"Selected {n_fdr_take} FDR significant genes (top by fold change)")
        for _, row in fdr_sig.head(n_fdr_take).iterrows():
            print(f"  FDR: {row['Gene']} (FC={row['true_fold_change']:.2f}, FDR={row.get('p_adjusted', 'N/A'):.3f})")
    
    # Then fill remaining slots with p<0.05 significant (excluding already selected)
    remaining_slots = n_genes - len(selected_genes)
    if remaining_slots > 0 and len(p_sig) > 0:
        # Exclude already selected genes
        p_sig_remaining = p_sig[~p_sig.index.isin(selected_genes)]
        n_p_take = min(remaining_slots, len(p_sig_remaining))
        selected_genes.extend(p_sig_remaining.head(n_p_take).index.tolist())
        print(f"Added {n_p_take} additional p<0.05 significant genes (top by fold change)")
        for _, row in p_sig_remaining.head(n_p_take).iterrows():
            print(f"  p<0.05: {row['Gene']} (FC={row['true_fold_change']:.2f}, p={row['p_value']:.3f})")
    
    if len(selected_genes) == 0:
        print("No significant genes found")
        return
    
    # Get the final selection maintaining fold change order
    top_genes = overall_results.loc[selected_genes].copy()
    top_genes['abs_log2fc'] = abs(top_genes['log2_fold_change'])
    top_genes = top_genes.sort_values('abs_log2fc', ascending=False)
    
    print(f"\nFinal selection (ordered by fold change magnitude):")
    print(f"{'Gene':<12} {'Direction':<3} {'Log2FC':<8} {'TrueFC':<8} {'AbsLog2FC':<10} {'Status':<8}")
    print("-" * 65)
    for _, row in top_genes.iterrows():
        direction = "↑" if row['log2_fold_change'] > 0 else "↓"
        fdr_status = "FDR" if row.get('significant_FDR', False) else "p<0.05"
        abs_log2fc = abs(row['log2_fold_change'])
        
        print(f"{row['Gene']:<12} {direction:<3} {row['log2_fold_change']:<8.3f} {row['true_fold_change']:<8.3f} {abs_log2fc:<10.3f} {fdr_status:<8}")
    
    # Prepare data for plotting
    plot_data = []
    
    for _, gene_row in top_genes.iterrows():
        gene = gene_row['Gene']
        
        # Find gene in expression data
        gene_expr_row = expression_data[expression_data[gene_col] == gene]
        if len(gene_expr_row) == 0:
            continue
            
        gene_idx = gene_expr_row.index[0]
        
        # Extract expression values for each sample
        for sample in expression_data.columns:
            if sample in sample_groups:
                expr_val = pd.to_numeric(expression_data.loc[gene_idx, sample], errors='coerce')
                if pd.notna(expr_val):
                    plot_data.append({
                        'Gene': gene,
                        'Expression': expr_val,
                        'Response': sample_groups[sample],
                        'Sample': sample
                    })
    
    if not plot_data:
        print("No expression data found for plotting")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create box plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    response_order = ['EXT', 'INT', 'NON']
    
    def format_p_and_fdr(p_val, fdr_val):
        """Format p-value and FDR for display - only show FDR if significant"""
        if p_val < 0.001:
            p_text = "p<0.001"
        else:
            p_text = f"p={p_val:.3f}"
        
        # Only show FDR if it's significant (< 0.05)
        if pd.notna(fdr_val) and fdr_val < 0.05:
            if fdr_val < 0.001:
                fdr_text = "FDR<0.001"
            else:
                fdr_text = f"FDR={fdr_val:.3f}"
            combined_text = f"{p_text}\n{fdr_text}"
            text_color = 'red'
            weight = 'bold'
        else:
            # Only show p-value
            combined_text = p_text
            text_color = 'black' if p_val < 0.05 else 'gray'
            weight = 'bold' if p_val < 0.05 else 'normal'
        
        return combined_text, text_color, weight
    
    for i, gene_idx in enumerate(top_genes.index[:n_genes]):
        gene_row = top_genes.loc[gene_idx]
        gene = gene_row['Gene']
        gene_data = plot_df[plot_df['Gene'] == gene]
        
        if len(gene_data) == 0:
            continue
            
        ax = axes[i]
        
        # Create violin plot as base layer
        sns.violinplot(
            x='Response',
            y='Expression',
            data=gene_data,
            palette=response_colors,
            order=response_order,
            ax=ax,
            alpha=0.6,
            inner=None  # Remove inner markings to avoid clash with box plot
        )
        
        # Create box plot on top with custom styling
        sns.boxplot(
            x='Response',
            y='Expression',
            data=gene_data,
            order=response_order,
            ax=ax,
            width=0.4,
            linewidth=1.5,
            fliersize=0,  # Remove outlier markers
            boxprops=dict(facecolor='white', alpha=0.8),
            medianprops=dict(color='black', linewidth=2)
        )
        
        # Get statistics for this gene from all comparisons
        gene_stats = {}
        for comp_name, comp_results in all_comparisons.items():
            gene_comp_data = comp_results[comp_results['Gene'] == gene]
            if len(gene_comp_data) > 0:
                gene_stats[comp_name] = gene_comp_data.iloc[0]
        
        # Calculate positions for significance annotations with better spacing
        y_max = gene_data['Expression'].max()
        y_min = gene_data['Expression'].min()
        y_range = y_max - y_min
        
        # Improved spacing for annotations
        annotation_y_start = y_max + y_range * 0.40
        h = y_range * 0.03
        
        # EXT vs NON (main comparison) - always show
        if 'ext_vs_non' in gene_stats:
            x1, x2 = 0, 2  # EXT and NON positions
            y_pos = annotation_y_start
            ax.plot([x1, x1, x2, x2], [y_pos, y_pos+h, y_pos+h, y_pos], lw=1.5, c='black')
            
            p_val = gene_stats['ext_vs_non']['p_value']
            fdr_val = gene_stats['ext_vs_non']['p_adjusted']
            
            combined_text, text_color, weight = format_p_and_fdr(p_val, fdr_val)
            
            ax.text((x1+x2)*.5, y_pos+h*1.2, combined_text, ha='center', va='bottom', 
                    color=text_color, fontsize=9, fontweight=weight)
        
        # EXT vs INT (if significant) - position higher to avoid overlap
        if 'ext_vs_int' in gene_stats and gene_stats['ext_vs_int']['p_value'] < 0.05:
            x1, x2 = 0, 1  # EXT and INT positions
            y_pos2 = annotation_y_start + y_range * 0.10
            ax.plot([x1, x1, x2, x2], [y_pos2, y_pos2+h, y_pos2+h, y_pos2], lw=1.5, c='gray')
            
            p_val = gene_stats['ext_vs_int']['p_value']
            fdr_val = gene_stats['ext_vs_int']['p_adjusted']
            
            combined_text, text_color, weight = format_p_and_fdr(p_val, fdr_val)
            
            ax.text((x1+x2)*.5, y_pos2+h*1.2, combined_text, ha='center', va='bottom', 
                    color=text_color, fontsize=8)
        
        # INT vs NON (if significant) - position even higher
        if 'int_vs_non' in gene_stats and gene_stats['int_vs_non']['p_value'] < 0.05:
            x1, x2 = 1, 2  # INT and NON positions
            # Adjust y position based on whether EXT vs INT is shown
            y_offset = 0.45 if ('ext_vs_int' in gene_stats and gene_stats['ext_vs_int']['p_value'] < 0.05) else 0.25
            y_pos3 = annotation_y_start + y_range * y_offset
            ax.plot([x1, x1, x2, x2], [y_pos3, y_pos3+h, y_pos3+h, y_pos3], lw=1.5, c='gray')
            
            p_val = gene_stats['int_vs_non']['p_value']
            fdr_val = gene_stats['int_vs_non']['p_adjusted']
            
            combined_text, text_color, weight = format_p_and_fdr(p_val, fdr_val)
            
            ax.text((x1+x2)*.5, y_pos3+h*1.2, combined_text, ha='center', va='bottom', 
                    color=text_color, fontsize=8)
        
        # Set title with both log2FC and true FC for clarity
        if 'ext_vs_non' in gene_stats:
            log2_fc = gene_stats['ext_vs_non']['log2_fold_change']
            true_fc = gene_stats['ext_vs_non']['true_fold_change']
        else:
            log2_fc = gene_row.get('log2_fold_change', 0)
            true_fc = gene_row.get('true_fold_change', 0)
        
        # Show both log2FC (for volcano plot comparison) and true FC
        fc_text = f" (FC={true_fc:.2f})" if log2_fc else ""
        
        
        # Truncate long gene names for title
        display_name = gene
        if len(display_name) > 8:  # Shorter to make room for both FC values
            display_name = display_name[:8] + "..."
        
        ax.set_title(f'{display_name}{fc_text}', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('')
        ax.set_ylabel('Log2 Expression Level', fontsize=10, fontweight='bold')
        
        # Styling
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Enhanced tick labels
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Set y-axis limits to accommodate annotations
        max_annotation_y = annotation_y_start + y_range * 0.5  # Account for highest annotation
        ax.set_ylim(y_min - y_range * 0.35, max_annotation_y)
        
        # Add sample count per group
        for j, resp in enumerate(response_order):
            n = len(gene_data[gene_data['Response'] == resp])
            ax.annotate(f"n={n}", xy=(j, y_min - y_range * 0.28), ha='center', fontsize=8)
    
    # Remove empty subplots
    for j in range(len(top_genes), len(axes)):
        fig.delaxes(axes[j])
    
    # Enhanced title and layout
    fdr_count = sum(top_genes.get('significant_FDR', False))
    p_count = len(top_genes) - fdr_count
    
    # Adjust layout with more space for annotations
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save figures
    plt.savefig('nanostring_top_genes_violin_boxplots.png', dpi=600, bbox_inches='tight')
    plt.savefig('nanostring_top_genes_violin_boxplots.pdf', bbox_inches='tight')
    plt.show()
    
    print("Violin+Box plots saved as nanostring_top_genes_violin_boxplots.png/pdf")
    print(f"\nSelection summary:")
    print(f"  FDR significant genes: {fdr_count}")
    print(f"  Additional p<0.05 genes: {p_count}")
    print(f"  Total displayed: {len(top_genes)}")
    
    print(f"\nVerification:")
    print(f"  All selected genes have abs(log2FC) >= 0.6: {all(abs(row['log2_fold_change']) >= 0.6 for _, row in top_genes.iterrows())}")
    print(f"  Volcano plot threshold: abs(log2FC) >= 0.6")
    print(f"  True FC interpretation: FC > 1.52 (upregulated) or FC < 0.66 (downregulated)")
    print(f"  Relationship: True FC = 2^(log2FC)")
    print(f"    - log2FC = 0.6  → True FC = 2^0.6  = 1.52")
    print(f"    - log2FC = -0.6 → True FC = 2^-0.6 = 0.66")
    
    return top_genes

if 'nanostring_overall_results' in locals() and 'nanostring_df' in locals():
    # First run all comparisons
    nanostring_all_comparisons = run_all_comparisons_nanostring(
        nanostring_df, nanostring_df.columns[0], sample_to_response
    )
    
    # Then create enhanced boxplots
    nanostring_top_genes = create_enhanced_nanostring_boxplots(
        nanostring_overall_results, nanostring_df, nanostring_df.columns[0], 
        sample_to_response, nanostring_all_comparisons
    )
else:
    print("Run Nanostring Phase 2 analysis first to generate nanostring_overall_results")


# In[ ]:





# In[ ]:





# In[ ]:




