#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

"""
Olink Phase 1: Data Verification & Diagnostic Analysis
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

print("OLINK PHASE 1: DATA VERIFICATION & DIAGNOSTIC ANALYSIS")
print("="*60)

# Load data
data_file = 'CICPT_1558_data (2).xlsx'

try:
    sheets = pd.read_excel(data_file, sheet_name=None)
    print(f"Loaded sheets: {list(sheets.keys())}")
    
    sample_info_df = sheets['sampleID']
    olink_df = sheets['Olink']
    nanostring_df = sheets['nanostring']  # Keep for reference
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Examine Olink data structure
print("\nOLINK DATA STRUCTURE ANALYSIS:")
print("-"*40)
print("Olink sheet shape:", olink_df.shape)
print("\nFirst 10 rows and columns:")
print(olink_df.iloc[:10, :5])

print("\nMetadata rows content:")
for i in range(min(4, len(olink_df))):
    print(f"Row {i}: {olink_df.iloc[i, :5].tolist()}")

# Identify the structure - check if data starts from row 3 (after metadata)
metadata_rows = 3  # Assay, UniProt ID, OlinkID
print(f"\nAssuming {metadata_rows} metadata rows")

# Extract protein information
if len(olink_df) > metadata_rows:
    protein_info = {}
    for col_idx in range(1, min(10, olink_df.shape[1])):  # Skip first column, check first few protein columns
        col_name = olink_df.columns[col_idx] if col_idx < len(olink_df.columns) else f"Col_{col_idx}"
        
        panel = olink_df.iloc[0, 0] if len(olink_df) > 0 else "Unknown"  # Panel info is in first column
        assay = olink_df.iloc[0, col_idx] if len(olink_df) > 0 else "Unknown"  # Row 0 contains Assay names
        uniprot = olink_df.iloc[1, col_idx] if len(olink_df) > 1 else "Unknown"  # Row 1 contains UniProt IDs
        olink_id = olink_df.iloc[2, col_idx] if len(olink_df) > 2 else "Unknown"  # Row 2 contains Olink IDs
        
        protein_info[col_name] = {
            'panel': panel,
            'assay': assay, 
            'uniprot': uniprot,
            'olink_id': olink_id
        }
    
    print("\nSample protein information:")
    for col, info in list(protein_info.items())[:3]:
        print(f"  {col}: Panel={info['panel']}, Assay={info['assay']}, UniProt={info['uniprot']}")

# Extract sample information from first column (starting after metadata rows)
sample_col = olink_df.iloc[metadata_rows:, 0]  # First column, after metadata
print(f"\nSample information (first 10):")
for i, sample in enumerate(sample_col.head(10)):
    print(f"  Row {metadata_rows + i}: {sample}")

# Identify control samples
control_samples = []
for i, sample in enumerate(sample_col):
    if pd.notna(sample) and 'CONTROL' in str(sample).upper():
        control_samples.append(sample)

print(f"\nIdentified {len(control_samples)} control samples:")
for ctrl in control_samples[:5]:  # Show first 5
    print(f"  {ctrl}")

def verify_olink_transformation():
    """Verify Olink NPX data scaling"""
    print("\nVERIFYING OLINK NPX DATA SCALING:")
    print("-"*40)
    
    # Get expression columns (excluding first column which has sample IDs)
    expr_cols = list(range(1, min(10, olink_df.shape[1])))  # First 9 protein columns
    
    all_values = []
    
    # Extract expression values from data rows (after metadata)
    for col_idx in expr_cols:
        if col_idx < olink_df.shape[1]:
            col_values = olink_df.iloc[metadata_rows:, col_idx]  # Skip metadata rows
            numeric_values = pd.to_numeric(col_values, errors='coerce').dropna()
            if len(numeric_values) > 0:
                all_values.extend(numeric_values.tolist())
    
    if len(all_values) > 0:
        all_values = np.array(all_values)
        print(f"Olink expression range: {all_values.min():.2f} to {all_values.max():.2f}")
        print(f"Mean: {all_values.mean():.2f}, SD: {all_values.std():.2f}")
        
        # NPX values are typically log2-scaled and centered around 0-15
        if -5 <= all_values.mean() <= 15 and 0.5 <= all_values.std() <= 5:
            print("✓ OLINK: NPX units detected (log2-scale)")
        else:
            print("⚠ OLINK: Unexpected value range - verify scaling")
    else:
        print("⚠ Could not extract expression values - check data format")

verify_olink_transformation()

# Create sample mappings using the same approach as Nanostring
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

# Map Olink samples to responses
olink_sample_col = olink_df.iloc[metadata_rows:, 0]  # Sample IDs from first column

# Create mapping from sample info
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
                
                # Extract timepoint from sample
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

# Also map Olink format samples (01-027B2W2 format)
for sample_id in olink_sample_col:
    if pd.notna(sample_id) and 'CONTROL' not in str(sample_id).upper():
        sample_str = str(sample_id)
        
        # Extract patient ID and timepoint from format like "01-027B2W2"
        match = re.search(r'(\d+-\d+)B(\d+)W(\d+)', sample_str)
        if match:
            patient_part = match.group(1)  # "01-027"
            block = int(match.group(2))
            week = int(match.group(3))
            timepoint = f"B{block}W{week}"
            
            # Find MRN for this patient
            patient_row = sample_info_df[sample_info_df['Study ID'] == patient_part]
            if len(patient_row) > 0:
                mrn = patient_row['MRN'].iloc[0]
                response = mrn_to_response.get(mrn)
                
                if response:
                    sample_to_response[sample_str] = response
                    sample_to_patientid[sample_str] = patient_part
                    
                    if block == 1:
                        phase = "Pembrolizumab_Mono_1"
                    elif block in [2, 3]:
                        phase = "Pembrolizumab_IL2"
                    elif block == 4:
                        phase = "Pembrolizumab_Mono_2"
                    else:
                        phase = "Unknown"
                    
                    sample_to_timepoint[sample_str] = {
                        'timepoint': timepoint,
                        'block': block,
                        'week': week,
                        'phase': phase
                    }

print(f"Mapped Olink samples: {len(sample_to_response)}")

response_counts = {}
for response in sample_to_response.values():
    response_counts[response] = response_counts.get(response, 0) + 1

print("Olink sample distribution:")
for response, count in sorted(response_counts.items()):
    print(f"  {response}: {count}")

unique_patients = len(set(sample_to_patientid.values()))
print(f"Unique patients in Olink: {unique_patients}")

# Identify which samples from Olink are actually present in our data
olink_samples_present = []
for sample_id in olink_sample_col:
    if pd.notna(sample_id) and str(sample_id) in sample_to_response:
        olink_samples_present.append(str(sample_id))

print(f"Olink samples present in response mapping: {len(olink_samples_present)}")

# Diagnostic fold change comparison (adapted for Olink)
print("\nOLINK FOLD CHANGE COMPARISON:")
print("-"*30)

# We need to restructure the Olink data for analysis
# Create a proper dataframe with proteins as rows and samples as columns
print("Restructuring Olink data for analysis...")

# Extract protein identifiers (using Assay names as primary identifier)
protein_identifiers = []
for col_idx in range(1, olink_df.shape[1]):
    if col_idx < olink_df.shape[1]:
        assay_name = olink_df.iloc[0, col_idx]  # Row 0 contains Assay names
        if pd.notna(assay_name):
            protein_identifiers.append(assay_name)

print(f"Found {len(protein_identifiers)} proteins in Olink data")

# Create restructured dataframe: proteins as rows, samples as columns
olink_restructured = pd.DataFrame()

# Add protein identifier column
olink_restructured['Protein'] = protein_identifiers

# Add sample expression columns
sample_row_indices = {}
for i, sample_id in enumerate(olink_sample_col):
    if pd.notna(sample_id):
        sample_str = str(sample_id)
        sample_row_indices[sample_str] = metadata_rows + i

# Add expression data for each sample
for sample_str, row_idx in sample_row_indices.items():
    if sample_str in sample_to_response:  # Only include mapped samples
        expr_values = []
        for col_idx in range(1, min(len(protein_identifiers) + 1, olink_df.shape[1])):
            expr_val = olink_df.iloc[row_idx, col_idx]
            expr_values.append(pd.to_numeric(expr_val, errors='coerce'))
        
        if len(expr_values) == len(protein_identifiers):
            olink_restructured[sample_str] = expr_values

print(f"Restructured Olink data shape: {olink_restructured.shape}")
print(f"Samples included: {olink_restructured.shape[1] - 1}")  # -1 for Protein column

# Quick fold change test with restructured data
if len(olink_restructured) > 0:
    expr_cols = [col for col in olink_restructured.columns if col != 'Protein' and col in sample_to_response]
    ext_samples = [col for col in expr_cols if sample_to_response[col] == 'EXT']
    non_samples = [col for col in expr_cols if sample_to_response[col] == 'NON']
    
    print(f"Expression columns for analysis: {len(expr_cols)}")
    print(f"EXT: {len(ext_samples)}, NON: {len(non_samples)}")
    
    if len(ext_samples) >= 3 and len(non_samples) >= 3:
        print("\n✓ Sufficient samples for differential analysis")
        
        # Test with first few proteins
        test_proteins = protein_identifiers[:5]
        
        print("\nProtein\t\tOLD Log2FC\tNEW Log2FC\tOLD FC\t\tNEW FC")
        print("-"*65)
        
        for protein in test_proteins:
            protein_rows = olink_restructured[olink_restructured['Protein'] == protein]
            if len(protein_rows) == 0:
                continue
            
            protein_idx = protein_rows.index[0]
            
            ext_vals = []
            non_vals = []
            
            for col in ext_samples:
                if col in olink_restructured.columns:
                    val = pd.to_numeric(olink_restructured.loc[protein_idx, col], errors='coerce')
                    if pd.notna(val):
                        ext_vals.append(val)
            
            for col in non_samples:
                if col in olink_restructured.columns:
                    val = pd.to_numeric(olink_restructured.loc[protein_idx, col], errors='coerce')
                    if pd.notna(val):
                        non_vals.append(val)
            
            if len(ext_vals) >= 3 and len(non_vals) >= 3:
                mean_ext = np.mean(ext_vals)
                mean_non = np.mean(non_vals)
                
                # OLD (incorrect for log2 data)
                old_fc = mean_ext / mean_non if mean_non > 0 else np.nan
                old_log2fc = np.log2(old_fc) if old_fc > 0 else np.nan
                
                # NEW (correct for log2 data)
                new_log2fc = mean_ext - mean_non
                new_fc = 2 ** new_log2fc
                
                protein_short = protein[:12] if len(protein) > 12 else protein
                print(f"{protein_short:<12}\t{old_log2fc:>7.3f}\t{new_log2fc:>7.3f}\t{old_fc:>7.3f}\t{new_fc:>7.3f}")

print("\n" + "="*60)
print("OLINK PHASE 1 COMPLETE")
print("="*60)

# Store for Phase 2
olink_phase1_data = {
    'sheets': sheets,
    'olink_restructured': olink_restructured,
    'sample_to_response': sample_to_response,
    'sample_to_patientid': sample_to_patientid,
    'sample_to_timepoint': sample_to_timepoint,
    'protein_identifiers': protein_identifiers,
    'control_samples': control_samples
}


# In[3]:


#!/usr/bin/env python
# coding: utf-8

"""
Olink Phase 2: Analysis 
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
response_colors = {'EXT': '#2ca02c', 'INT': '#1f77b4', 'NON': '#d62728'}

print("OLINK PHASE 2: ANALYSIS")
print("="*60)

# Load data from Phase 1
if 'olink_phase1_data' not in locals():
    print("Run Olink Phase 1 first!")
    exit()

sheets = olink_phase1_data['sheets']
olink_restructured = olink_phase1_data['olink_restructured']
sample_to_response = olink_phase1_data['sample_to_response']
sample_to_patientid = olink_phase1_data['sample_to_patientid']
sample_to_timepoint = olink_phase1_data['sample_to_timepoint']
protein_identifiers = olink_phase1_data['protein_identifiers']
control_samples = olink_phase1_data['control_samples']

def is_control_protein(protein_name):
    """Identify control proteins to exclude"""
    if pd.isna(protein_name):
        return True
    protein_str = str(protein_name).upper()
    controls = ['CONTROL', 'NEGATIVE', 'BLANK', 'BUFFER', 'SPIKE']
    return any(ctrl in protein_str for ctrl in controls)

def robust_differential_analysis_corrected_olink(data, protein_col, group1, group2, sample_groups, min_samples=3):
    """Corrected differential analysis for log2 Olink NPX data"""
    
    group1_cols = [col for col in data.columns if sample_groups.get(col) == group1]
    group2_cols = [col for col in data.columns if sample_groups.get(col) == group2]
    
    if len(group1_cols) < min_samples or len(group2_cols) < min_samples:
        return None
    
    results = []
    
    for idx, row in data.iterrows():
        protein = row[protein_col]
        
        # Skip control proteins
        if is_control_protein(protein):
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
        
        # calculations for log2 NPX data
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
            'Protein': protein,
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

def create_volcano_plot_formatted_olink(data, title="Olink Volcano Plot", save_name="olink_volcano_plot"):
    """Create volcano plot with exact formatting from provided code"""
    
    print("CREATING OLINK VOLCANO PLOT")
    print("="*50)
    
    # Remove any rows with missing data
    data = data.dropna(subset=['log2_fold_change', 'p_value'])
    
    # Exclude controls (already done but double-check)
    data = data[~data['Protein'].str.contains('CONTROL|NEGATIVE|BLANK', case=False, na=False)]
    
    # Define thresholds
    FC_THRESHOLD = 0.6  # log2(1.5)
    P_THRESHOLD = 0.05
    
    print(f"Thresholds: Log2FC ±{FC_THRESHOLD}, p-value {P_THRESHOLD}")
    
    # Create significance categories
    data['meets_fc_threshold'] = abs(data['log2_fold_change']) >= FC_THRESHOLD
    data['meets_p_threshold'] = data['p_value'] < P_THRESHOLD
    data['meets_both_criteria'] = data['meets_fc_threshold'] & data['meets_p_threshold']
    
    sig_both = data[data['meets_both_criteria']]
    
    print(f"Total proteins: {len(data)}")
    print(f"Meeting both criteria: {len(sig_both)}")
    
    # CREATE VOLCANO PLOT
    plt.figure(figsize=(10, 8))
    
    # Plot all points (base layer)
    plt.scatter(
        data['log2_fold_change'],
        -np.log10(data['p_value']),
        alpha=0.6,
        s=40,
        color='#cccccc',
        edgecolor='none',
        zorder=1
    )
    
    # Plot significant points (red)
    if len(sig_both) > 0:
        plt.scatter(
            sig_both['log2_fold_change'],
            -np.log10(sig_both['p_value']),
            alpha=0.9,
            s=60,
            color='#e41a1c',
            edgecolor='black',
            linewidth=0.5,
            zorder=2
        )
    
    # Add threshold lines
    plt.axhline(y=-np.log10(P_THRESHOLD), color='#e41a1c', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
    plt.axvline(x=FC_THRESHOLD, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
    plt.axvline(x=-FC_THRESHOLD, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
    
    # Add p-value threshold label
    plt.text(
        0.35, 
        -np.log10(P_THRESHOLD) + 0.05, 
        f'p = {P_THRESHOLD}', 
        color='#e41a1c',
        fontsize=10,
        fontstyle='italic',
        verticalalignment='bottom'
    )
    
    # Label top significant proteins
    if len(sig_both) > 0:
        top_sig = sig_both.nsmallest(min(20, len(sig_both)), 'p_value')
        
        for _, row in top_sig.iterrows():
            # Truncate long protein names for better display
            protein_name = row['Protein']
            if len(protein_name) > 10:
                protein_name = protein_name[:10] + "..."
            
            plt.text(
                row['log2_fold_change'], 
                -np.log10(row['p_value']), 
                protein_name,
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='bottom',
                zorder=3
            )
    
    # Axis labels and styling
    plt.xlabel('Log2 Fold Change (EXT vs NON)', fontsize=14, fontweight='bold')
    plt.ylabel('-Log10(p-value)', fontsize=14, fontweight='bold')
    
    plt.grid(alpha=0.3, linestyle='--', zorder=0)
    
    # Set axis limits
    x_range = max(abs(data['log2_fold_change'].min()), abs(data['log2_fold_change'].max()))
    y_max = max(-np.log10(data['p_value']))
    
    plt.xlim(-x_range * 1.3, x_range * 1.3)
    plt.ylim(-0.1, y_max * 1.2)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
    
    plt.show()
    
    return sig_both

def create_heatmap_formatted_olink(results_df, expression_data, protein_col, sample_groups, save_name="olink_heatmap"):
    """Create heatmap with exact formatting for Olink data"""
    
    sig_proteins = results_df[results_df['p_value'] < 0.05]['Protein'].tolist()
    
    if len(sig_proteins) == 0:
        print("No significant proteins for heatmap")
        return
    
    # Calculate mean expression for each group
    expression_matrix = []
    protein_names = []
    
    for protein in sig_proteins[:50]:  # Top 50
        protein_row = expression_data[expression_data[protein_col] == protein]
        if len(protein_row) == 0:
            continue
            
        protein_idx = protein_row.index[0]
        means = []
        
        for response in ['EXT', 'INT', 'NON']:
            samples = [s for s in expression_data.columns if sample_groups.get(s) == response]
            if samples:
                values = []
                for col in samples:
                    if col in expression_data.columns:
                        val = pd.to_numeric(expression_data.loc[protein_idx, col], errors='coerce')
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
            protein_names.append(protein)
    
    if expression_matrix:
        expression_matrix = np.array(expression_matrix)
        
        # Z-score normalize each row
        expression_matrix_z = np.zeros_like(expression_matrix)
        for i in range(len(protein_names)):
            row = expression_matrix[i, :]
            if np.std(row) > 0:
                expression_matrix_z[i, :] = (row - np.mean(row)) / np.std(row)
            else:
                expression_matrix_z[i, :] = row - np.mean(row)
        
        plt.figure(figsize=(7, max(8, len(protein_names) * 0.4)))
        im = plt.imshow(expression_matrix_z, aspect='auto', cmap='RdBu_r', interpolation='none')
        
        cbar = plt.colorbar(im)
        cbar.set_label('Mean Protein Expression (Z-score)', fontsize=12, fontweight='bold')
        
        plt.yticks(np.arange(len(protein_names)), protein_names, fontsize=10)
        plt.xticks(np.arange(3), ['EXT', 'INT', 'NON'], fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
        plt.show()


# MAIN ANALYSIS EXECUTION

print("1. Overall Differential Analysis (EXT vs NON)")
print("-"*40)

overall_results = robust_differential_analysis_corrected_olink(
    olink_restructured, 'Protein', 'EXT', 'NON', sample_to_response
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
    
    print(f"Analyzed {len(overall_results)} proteins (controls excluded)")
    print(f"Significant (p < 0.05): {n_sig_raw}")
    print(f"Significant (FDR < 0.05): {n_sig_fdr}")
    
    # Save results
    overall_results.to_csv('olink_corrected_comprehensive.csv', index=False)
    
    # Create formatted plots
    sig_proteins = create_volcano_plot_formatted_olink(
        overall_results, 
        "Olink Protein Differential Expression (Corrected)", 
        "olink_volcano_corrected"
    )
    
    create_heatmap_formatted_olink(
        overall_results, olink_restructured, 'Protein', 
        sample_to_response, "olink_heatmap_corrected"
    )
    
    # Top significant proteins
    print(f"\nTop significant proteins:")
    for _, row in overall_results.head(25).iterrows():
        direction = "↑" if row['log2_fold_change'] > 0 else "↓"
        print(f"  {row['Protein']} {direction} (log2FC={row['log2_fold_change']:.2f}, "
              f"FC={row['true_fold_change']:.2f}, p={row['p_value']:.2e})")

print("\n" + "="*60)
print("OLINK PHASE 2 COMPLETE")
print("="*60)

print("\nFiles generated:")
print("  - olink_corrected_comprehensive.csv")
print("  - olink_volcano_corrected.png/pdf")
print("  - olink_heatmap_corrected.png/pdf")

# Store results for next phase
olink_phase2_data = {
    'overall_results': overall_results,
    'olink_restructured': olink_restructured,
    'sample_to_response': sample_to_response,
    'sample_to_patientid': sample_to_patientid,
    'sample_to_timepoint': sample_to_timepoint
}


# In[4]:


#!/usr/bin/env python
# coding: utf-8

"""
Olink Phase 3: Comprehensive Comparisons and Box Plots
All three pairwise comparisons with detailed visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
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
response_colors = {'EXT': '#2ca02c', 'INT': '#1f77b4', 'NON': '#d62728'}

print("OLINK PHASE 3: COMPREHENSIVE COMPARISONS AND BOX PLOTS")
print("="*60)

def is_control_protein(protein_name):
    """Identify control proteins to exclude"""
    if pd.isna(protein_name):
        return True
    protein_str = str(protein_name).upper()
    controls = ['CONTROL', 'NEGATIVE', 'BLANK', 'BUFFER', 'SPIKE']
    return any(ctrl in protein_str for ctrl in controls)

def robust_differential_analysis_corrected_olink(data, protein_col, group1, group2, sample_groups, min_samples=3):
    """Corrected differential analysis for log2 Olink NPX data"""
    
    group1_cols = [col for col in data.columns if sample_groups.get(col) == group1]
    group2_cols = [col for col in data.columns if sample_groups.get(col) == group2]
    
    if len(group1_cols) < min_samples or len(group2_cols) < min_samples:
        return None
    
    results = []
    
    for idx, row in data.iterrows():
        protein = row[protein_col]
        
        # Skip control proteins
        if is_control_protein(protein):
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
        
        # calculations for log2 NPX data
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
            'Protein': protein,
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

def run_all_comparisons_olink(expression_data, protein_col, sample_groups):
    """Run all three comparisons: EXT vs NON, EXT vs INT, INT vs NON"""
    
    print("Running all pairwise comparisons for Olink proteins...")
    
    comparisons = [
        ('EXT', 'NON', 'ext_vs_non'),
        ('EXT', 'INT', 'ext_vs_int'), 
        ('INT', 'NON', 'int_vs_non')
    ]
    
    all_results = {}
    
    for group1, group2, comp_name in comparisons:
        print(f"\nAnalyzing {group1} vs {group2}...")
        
        results = robust_differential_analysis_corrected_olink(
            expression_data, protein_col, group1, group2, sample_groups
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
            results.to_csv(f'olink_{comp_name}_corrected.csv', index=False)
    
    return all_results

def create_boxplots_top_proteins_olink(results_df, expression_data, protein_col, sample_groups, all_comparisons, n_proteins=6):
    """Create box plots with all comparisons for Olink proteins"""
    
    print("Creating box plots for top significant proteins...")
    
    # prioritize FDR significant, then p<0.05, all sorted by fold change
    fdr_sig = results_df[results_df.get('significant_FDR', pd.Series(False, index=results_df.index))].copy()
    p_sig = results_df[results_df.get('significant_raw', results_df['p_value'] < 0.05)].copy()
    
    # Calculate absolute fold change for sorting
    if len(fdr_sig) > 0:
        fdr_sig['abs_log2fc'] = abs(fdr_sig['log2_fold_change'])
        fdr_sig = fdr_sig.sort_values('abs_log2fc', ascending=False)
    
    if len(p_sig) > 0:
        p_sig['abs_log2fc'] = abs(p_sig['log2_fold_change'])
        p_sig = p_sig.sort_values('abs_log2fc', ascending=False)
    
    print(f"Available proteins: {len(results_df)} total")
    print(f"FDR significant available: {len(fdr_sig)}")
    print(f"p<0.05 significant available: {len(p_sig)}")
    
    # Select top proteins: FDR first, then p<0.05 to fill remaining slots
    selected_proteins = []
    
    # First, take FDR significant proteins (up to n_proteins)
    if len(fdr_sig) > 0:
        n_fdr_take = min(n_proteins, len(fdr_sig))
        selected_proteins.extend(fdr_sig.head(n_fdr_take).index.tolist())
        print(f"Selected {n_fdr_take} FDR significant proteins (top by fold change)")
        for _, row in fdr_sig.head(n_fdr_take).iterrows():
            print(f"  FDR: {row['Protein']} (FC={row['true_fold_change']:.2f}, FDR={row.get('p_adjusted', 'N/A'):.3f})")
    
    # Then fill remaining slots with p<0.05 significant (excluding already selected)
    remaining_slots = n_proteins - len(selected_proteins)
    if remaining_slots > 0 and len(p_sig) > 0:
        # Exclude already selected proteins
        p_sig_remaining = p_sig[~p_sig.index.isin(selected_proteins)]
        n_p_take = min(remaining_slots, len(p_sig_remaining))
        selected_proteins.extend(p_sig_remaining.head(n_p_take).index.tolist())
        print(f"Added {n_p_take} additional p<0.05 significant proteins (top by fold change)")
        for _, row in p_sig_remaining.head(n_p_take).iterrows():
            print(f"  p<0.05: {row['Protein']} (FC={row['true_fold_change']:.2f}, p={row['p_value']:.3f})")
    
    if len(selected_proteins) == 0:
        print("No significant proteins found")
        return
    
    # Get the final selection maintaining fold change order
    top_proteins = results_df.loc[selected_proteins].copy()
    top_proteins['abs_log2fc'] = abs(top_proteins['log2_fold_change'])
    top_proteins = top_proteins.sort_values('abs_log2fc', ascending=False)
    
    print(f"\nFinal selection (ordered by fold change magnitude):")
    print(f"{'Protein':<12} {'Direction':<3} {'Log2FC':<8} {'TrueFC':<8} {'AbsLog2FC':<10} {'Status':<8}")
    print("-" * 65)
    for _, row in top_proteins.iterrows():
        direction = "↑" if row['log2_fold_change'] > 0 else "↓"
        fdr_status = "FDR" if row.get('significant_FDR', False) else "p<0.05"
        fdr_val = row.get('p_adjusted', np.nan)
        fdr_text = f", FDR={fdr_val:.3f}" if pd.notna(fdr_val) else ""
        abs_log2fc = abs(row['log2_fold_change'])
        
        print(f"{row['Protein']:<12} {direction:<3} {row['log2_fold_change']:<8.3f} {row['true_fold_change']:<8.3f} {abs_log2fc:<10.3f} {fdr_status:<8}")
    
    print(f"\nVerification:")
    print(f"  All selected proteins have abs(log2FC) >= 0.6: {all(abs(row['log2_fold_change']) >= 0.6 for _, row in top_proteins.iterrows())}")
    print(f"  Volcano plot threshold: abs(log2FC) >= 0.6")
    print(f"  True FC interpretation: FC > 1.52 (upregulated) or FC < 0.66 (downregulated)")
    print(f"  Relationship: True FC = 2^(log2FC)")
    print(f"    - log2FC = 0.6  → True FC = 2^0.6  = 1.52")
    print(f"    - log2FC = -0.6 → True FC = 2^-0.6 = 0.66")
    
    # Prepare data for plotting
    plot_data = []
    
    for _, protein_row in top_proteins.iterrows():
        protein = protein_row['Protein']
        
        # Find protein in expression data
        protein_expr_row = expression_data[expression_data[protein_col] == protein]
        if len(protein_expr_row) == 0:
            continue
            
        protein_idx = protein_expr_row.index[0]
        
        # Extract expression values for each sample
        for sample in expression_data.columns:
            if sample in sample_groups:
                expr_val = pd.to_numeric(expression_data.loc[protein_idx, sample], errors='coerce')
                if pd.notna(expr_val):
                    plot_data.append({
                        'Protein': protein,
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
    
    for i, protein_idx in enumerate(top_proteins.index[:n_proteins]):
        protein_row = top_proteins.loc[protein_idx]
        protein = protein_row['Protein']
        protein_data = plot_df[plot_df['Protein'] == protein]
        
        if len(protein_data) == 0:
            continue
            
        ax = axes[i]
        
        # Create violin plot as base layer
        sns.violinplot(
            x='Response',
            y='Expression',
            data=protein_data,
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
            data=protein_data,
            order=response_order,
            ax=ax,
            width=0.4,
            linewidth=1.5,
            fliersize=0,  # Remove outlier markers
            boxprops=dict(facecolor='white', alpha=0.8),
            medianprops=dict(color='black', linewidth=2)
        )
        
        # Get statistics for this protein from all comparisons
        protein_stats = {}
        for comp_name, comp_results in all_comparisons.items():
            protein_comp_data = comp_results[comp_results['Protein'] == protein]
            if len(protein_comp_data) > 0:
                protein_stats[comp_name] = protein_comp_data.iloc[0]
        
        # Calculate positions for significance annotations
        y_max = protein_data['Expression'].max()
        y_min = protein_data['Expression'].min()
        y_range = y_max - y_min
        
        # Get statistics for this protein from all comparisons
        protein_stats = {}
        for comp_name, comp_results in all_comparisons.items():
            protein_comp_data = comp_results[comp_results['Protein'] == protein]
            if len(protein_comp_data) > 0:
                protein_stats[comp_name] = protein_comp_data.iloc[0]
        
        # Calculate positions for significance annotations with better spacing
        y_max = protein_data['Expression'].max()
        y_min = protein_data['Expression'].min()
        y_range = y_max - y_min
        
        # Improved spacing for annotations
        annotation_y_start = y_max + y_range * 0.30
        h = y_range * 0.03
        
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
        
        # EXT vs NON (main comparison)
        if 'ext_vs_non' in protein_stats:
            x1, x2 = 0, 2  # EXT and NON positions
            y_pos = annotation_y_start
            ax.plot([x1, x1, x2, x2], [y_pos, y_pos+h, y_pos+h, y_pos], lw=1.5, c='black')
            
            p_val = protein_stats['ext_vs_non']['p_value']
            fdr_val = protein_stats['ext_vs_non']['p_adjusted']
            
            combined_text, text_color, weight = format_p_and_fdr(p_val, fdr_val)
            
            ax.text((x1+x2)*.5, y_pos+h*1.2, combined_text, ha='center', va='bottom', 
                    color=text_color, fontsize=9, fontweight=weight)
        
        # EXT vs INT (if significant)
        if 'ext_vs_int' in protein_stats and protein_stats['ext_vs_int']['p_value'] < 0.05:
            x1, x2 = 0, 1  # EXT and INT positions
            y_pos2 = annotation_y_start + y_range * 0.15
            ax.plot([x1, x1, x2, x2], [y_pos2, y_pos2+h, y_pos2+h, y_pos2], lw=1.5, c='gray')
            
            p_val = protein_stats['ext_vs_int']['p_value']
            fdr_val = protein_stats['ext_vs_int']['p_adjusted']
            
            combined_text, text_color, weight = format_p_and_fdr(p_val, fdr_val)
            
            ax.text((x1+x2)*.5, y_pos2+h*1.2, combined_text, ha='center', va='bottom', 
                    color=text_color, fontsize=8)
        
        # INT vs NON (if significant)
        if 'int_vs_non' in protein_stats and protein_stats['int_vs_non']['p_value'] < 0.05:
            x1, x2 = 1, 2  # INT and NON positions
            # Adjust y position based on whether EXT vs INT is shown
            y_offset = 0.45 if ('ext_vs_int' in protein_stats and protein_stats['ext_vs_int']['p_value'] < 0.05) else 0.25
            y_pos3 = annotation_y_start + y_range * y_offset
            ax.plot([x1, x1, x2, x2], [y_pos3, y_pos3+h, y_pos3+h, y_pos3], lw=1.5, c='gray')
            
            p_val = protein_stats['int_vs_non']['p_value']
            fdr_val = protein_stats['int_vs_non']['p_adjusted']
            
            combined_text, text_color, weight = format_p_and_fdr(p_val, fdr_val)
            
            ax.text((x1+x2)*.5, y_pos3+h*1.2, combined_text, ha='center', va='bottom', 
                    color=text_color, fontsize=8)
        
        # Set title with both log2FC and true FC
        if 'ext_vs_non' in protein_stats:
            log2_fc = protein_stats['ext_vs_non']['log2_fold_change']
            true_fc = protein_stats['ext_vs_non']['true_fold_change']
        else:
            log2_fc = protein_row.get('log2_fold_change', 0)
            true_fc = protein_row.get('true_fold_change', 0)
        
        # Show both log2FC (for volcano plot comparison) and true FC
        fc_text = f" (FC={true_fc:.2f})" if log2_fc else ""
        
        # Truncate long protein names for title
        display_name = protein
        if len(display_name) > 8:  # Shorter to make room for both FC values
            display_name = display_name[:8] + "..."
        
        ax.set_title(f'{display_name}{fc_text}', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('')
        ax.set_ylabel('Log2 NPX Expression Level', fontsize=10, fontweight='bold')
        
        # Styling
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Enhanced tick labels
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Set y-axis limits to accommodate annotations
        max_annotation_y = annotation_y_start + y_range * 0.5  # Account for highest annotation
        ax.set_ylim(y_min - y_range * 0.30, max_annotation_y)
        
        # Add sample count per group
        for j, resp in enumerate(response_order):
            n = len(protein_data[protein_data['Response'] == resp])
            ax.annotate(f"n={n}", xy=(j, y_min - y_range * 0.28), ha='center', fontsize=8)
    
    # Remove empty subplots
    for j in range(len(top_proteins), len(axes)):
        fig.delaxes(axes[j])
    
    # Enhanced title and layout
    fdr_count = sum(top_proteins.get('significant_FDR', False))
    p_count = len(top_proteins) - fdr_count
    
    # Adjust layout with more space for annotations
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save figures
    plt.savefig('olink_top_proteins_violin_boxplots.png', dpi=600, bbox_inches='tight')
    plt.savefig('olink_top_proteins_violin_boxplots.pdf', bbox_inches='tight')
    plt.show()
    
    print("Violin+Box plots saved as olink_top_proteins_violin_boxplots.png/pdf")
    print(f"\nSelection summary:")
    print(f"  FDR significant proteins: {fdr_count}")
    print(f"  Additional p<0.05 proteins: {p_count}")
    print(f"  Total displayed: {len(top_proteins)}")
    
    return top_proteins

def check_fdr_significance_olink(all_comparisons):
    """Check which proteins are FDR significant in any comparison"""
    
    print("\nOLINK FDR SIGNIFICANCE ANALYSIS:")
    print("="*40)
    
    fdr_significant_proteins = set()
    
    for comp_name, results in all_comparisons.items():
        fdr_sig = results[results['significant_FDR']]
        if len(fdr_sig) > 0:
            fdr_significant_proteins.update(fdr_sig['Protein'].tolist())
            print(f"\n{comp_name.upper().replace('_', ' ')}:")
            print(f"  FDR significant proteins: {len(fdr_sig)}")
            
            for _, row in fdr_sig.head(5).iterrows():
                direction = "↑" if row['log2_fold_change'] > 0 else "↓"
                print(f"    {row['Protein']} {direction} (FDR={row['p_adjusted']:.3f}, FC={row['true_fold_change']:.2f})")
    
    print(f"\nTotal unique FDR significant proteins: {len(fdr_significant_proteins)}")
    
    if fdr_significant_proteins:
        print("FDR significant proteins:", sorted(fdr_significant_proteins))
    
    return fdr_significant_proteins

# MAIN EXECUTION
if 'olink_phase2_data' in locals():
    # Use data from Phase 2
    overall_results = olink_phase2_data['overall_results']
    olink_restructured = olink_phase2_data['olink_restructured']
    sample_to_response = olink_phase2_data['sample_to_response']
    sample_to_patientid = olink_phase2_data['sample_to_patientid']
    sample_to_timepoint = olink_phase2_data['sample_to_timepoint']
    
    # Run all comparisons
    all_comparisons = run_all_comparisons_olink(olink_restructured, 'Protein', sample_to_response)
    
    # Check FDR significance
    fdr_proteins = check_fdr_significance_olink(all_comparisons)
    
    # Create box plots with all comparisons
    top_proteins_plotted = create_boxplots_top_proteins_olink(
        overall_results, olink_restructured, 'Protein', 
        sample_to_response, all_comparisons
    )
    
    print("\n" + "="*60)
    print("OLINK PHASE 3 COMPLETE")
    print("="*60)
    
    print("\nFiles generated:")
    print("  - olink_ext_vs_non_corrected.csv")
    print("  - olink_ext_vs_int_corrected.csv") 
    print("  - olink_int_vs_non_corrected.csv")
    print("  - olink_top_proteins_boxplots.png/pdf")
    
    # Store for Phase 4
    olink_phase3_data = {
        'all_comparisons': all_comparisons,
        'fdr_proteins': fdr_proteins,
        'top_proteins_plotted': top_proteins_plotted,
        'olink_restructured': olink_restructured,
        'sample_to_response': sample_to_response,
        'sample_to_patientid': sample_to_patientid,
        'sample_to_timepoint': sample_to_timepoint
    }

else:
    print("Run Olink Phase 2 first to generate overall_results")


# In[ ]:




