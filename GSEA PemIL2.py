#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import matplotlib.patches as mpatches
import re
import os
import textwrap
import warnings
from statsmodels.stats.multitest import multipletests
import gseapy as gp
from gseapy import prerank

# Suppress warnings
warnings.filterwarnings("ignore")

# Set plot style
plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

print("=" * 80)
print("CORRECTED GSEA ANALYSIS - LOG2 TRANSFORMED DATA")
print("Using verified data loading approach")
print("=" * 80)

# Create output directories
os.makedirs('gsea_results_corrected', exist_ok=True)
os.makedirs('gsea_results_corrected/nanostring', exist_ok=True)
os.makedirs('gsea_results_corrected/olink', exist_ok=True)
os.makedirs('gsea_results_corrected/plots', exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load data"""
    
    print("Loading data...")
    data_file = 'CICPT_1558_data (2).xlsx'
    
    # --- ADDED DIAGNOSTIC CHECKS ---
    print(f"Attempting to load file: '{data_file}'")
    if not os.path.exists(data_file):
        print(f"Error: File '{data_file}' not found in the current directory: {os.getcwd()}")
        print("Please ensure the Excel file is uploaded to the same directory as your Jupyter Notebook.")
        return None, None, None, None
    print(f"File '{data_file}' found. Proceeding with pandas.read_excel...")
    # -------------------------------
    
    try:
        sheets = pd.read_excel(data_file, sheet_name=None)
        print(f"Successfully loaded sheets: {list(sheets.keys())}")
        
        # Check if expected sheets exist
        expected_sheets = ['sampleID', 'nanostring', 'Olink']
        for sheet_name in expected_sheets:
            if sheet_name not in sheets:
                print(f"Error: Required sheet '{sheet_name}' not found in '{data_file}'.")
                return None, None, None, None

        sample_info_df = sheets['sampleID']
        nanostring_df = sheets['nanostring']
        olink_df = sheets['Olink']
        
        print(f"Sample Info shape: {sample_info_df.shape}")
        print(f"Nanostring data shape: {nanostring_df.shape}")
        print(f"Olink data shape: {olink_df.shape}")

        return sheets, sample_info_df, nanostring_df, olink_df
        
    except Exception as e:
        print(f"Error loading data with pandas.read_excel: {e}")
        print("This error or a hang might occur if the Excel file is very large or corrupted.")
        print("Consider checking the file size and integrity.")
        return None, None, None, None

def create_sample_mappings(sheets, sample_info_df):
    """Create sample mappings using the verified approach"""
    
    print("\nCreating sample mappings using verified approach...")
    
    # Load response group data with header=None
    # This also re-reads the Excel file, ensure it's the correct path
    data_file = 'CICPT_1558_data (2).xlsx' # Ensure this is consistent
    try:
        response_df = pd.read_excel(data_file, sheet_name="group", header=None)
        print("Successfully loaded 'group' sheet.")
    except Exception as e:
        print(f"Error loading 'group' sheet from '{data_file}': {e}")
        return {}, {}, {}, {}
    
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
                except ValueError: # Catch specific error for int conversion
                    mrn_to_response[mrn] = response_group # Keep as string if not int
    
    print(f"MRN mappings: {len(mrn_to_response)}")
    print(f"Response groups: {set(mrn_to_response.values())}")
    
    # Create sample mappings
    sample_to_response = {}
    sample_to_patientid = {}
    sample_to_timepoint = {}
    
    # Map samples to responses
    for _, row in sample_info_df.iterrows():
        # Using .get() for safer access to columns
        mrn = row.get('MRN')
        study_id = row.get('Study ID')
        
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
    
    return sample_to_response, sample_to_patientid, sample_to_timepoint, mrn_to_response

def process_nanostring_data_corrected(nanostring_df, sample_to_response):
    """Process Nanostring data"""
    
    print("\nProcessing Nanostring data (corrected for log2)...")
    
    # Find expression columns that match our sample mappings
    expr_cols = [col for col in nanostring_df.columns if col in sample_to_response]
    print(f"Found {len(expr_cols)} expression columns matching sample mappings")
    
    if not expr_cols:
        print("Warning: No Nanostring expression columns found that match sample mappings. Check data consistency.")
        return pd.DataFrame()
        
    # Get gene column (first column)
    gene_col = nanostring_df.columns[0]
    print(f"Using '{gene_col}' as gene identifier")
    
    # Create expression matrix
    nano_expression = nanostring_df.set_index(gene_col)[expr_cols].copy()
    
    print(f"Nanostring expression matrix: {nano_expression.shape}")
    print(f"Genes: {nano_expression.shape[0]}, Samples: {nano_expression.shape[1]}")
    
    return nano_expression

def process_olink_data_corrected(olink_df, sample_info_df, sample_to_response, mrn_to_response):
    """Process Olink data using verified approach with proper restructuring"""
    
    print("\nProcessing Olink data (corrected for log2)...")
    
    # Olink data structure: 3 metadata rows, then samples
    metadata_rows = 3
    print(f"Assuming {metadata_rows} metadata rows in Olink data")
    
    # Extract protein information from metadata rows
    protein_identifiers = []
    # Iterate through columns, starting from 1 (assuming first col is sample ID)
    for col_idx in range(1, olink_df.shape[1]):
        assay_name = olink_df.iloc[0, col_idx]  # Row 0 contains Assay names
        if pd.notna(assay_name):
            protein_identifiers.append(str(assay_name))
    
    print(f"Found {len(protein_identifiers)} proteins in Olink data")
    
    # Extract sample information from first column (after metadata rows)
    # Ensure this slices the correct part for sample IDs
    sample_col_data = olink_df.iloc[metadata_rows:, 0]
    
    # Identify control samples to exclude
    control_samples = []
    for sample in sample_col_data:
        if pd.notna(sample) and 'CONTROL' in str(sample).upper():
            control_samples.append(sample)
    
    print(f"Identified {len(control_samples)} control samples to exclude")
    
    # Create extended sample mapping for Olink format
    olink_sample_to_response = sample_to_response.copy()
    
    # Map Olink format samples (01-027B2W2 format)
    for sample_id in sample_col_data:
        if pd.notna(sample_id) and 'CONTROL' not in str(sample_id).upper():
            sample_str = str(sample_id)
            
            # Extract patient ID and timepoint from format like "01-027B2W2"
            match = re.search(r'(\d+-\d+)B(\d+)W(\d+)', sample_str)
            if match:
                patient_part = match.group(1)  # "01-027"
                
                # Find MRN for this patient
                patient_row = sample_info_df[sample_info_df['Study ID'] == patient_part]
                if not patient_row.empty: # Check if patient_row is not empty
                    mrn = patient_row['MRN'].iloc[0]
                    response = mrn_to_response.get(mrn)
                    
                    if response:
                        olink_sample_to_response[sample_str] = response
    
    print(f"Extended Olink sample mappings: {len(olink_sample_to_response)}")
    
    # Restructure Olink data: proteins as rows, samples as columns
    olink_restructured = pd.DataFrame(index=protein_identifiers) # Initialize with proteins as index
    
    valid_olink_sample_columns = []
    
    # Populate the DataFrame with expression data
    for sample_str_from_olink, row_idx_in_olink_df in enumerate(olink_df.index[metadata_rows:], start=metadata_rows):
        sample_id = olink_df.iloc[row_idx_in_olink_df, 0] # Get the actual sample ID string
        
        if pd.notna(sample_id) and str(sample_id) in olink_sample_to_response and 'CONTROL' not in str(sample_id).upper():
            current_sample_id = str(sample_id)
            expr_values = []
            
            for col_idx in range(1, len(protein_identifiers) + 1): # Iterate through protein columns
                expr_val = olink_df.iloc[row_idx_in_olink_df, col_idx]
                expr_values.append(pd.to_numeric(expr_val, errors='coerce'))
            
            # Ensure the number of values matches the number of proteins before adding
            if len(expr_values) == len(protein_identifiers):
                olink_restructured[current_sample_id] = expr_values
                valid_olink_sample_columns.append(current_sample_id)

    # Ensure Olink expression only contains valid samples and proteins
    olink_expression = olink_restructured.loc[protein_identifiers, valid_olink_sample_columns]
    
    print(f"Olink expression matrix: {olink_expression.shape}")
    print(f"Proteins: {olink_expression.shape[0]}, Samples: {olink_expression.shape[1]}")
    print(f"Valid samples included in matrix: {len(valid_olink_sample_columns)}")
    
    return olink_expression, olink_sample_to_response

# ============================================================================
# DIFFERENTIAL EXPRESSION (CORRECTED FOR LOG2 DATA)
# ============================================================================

def perform_differential_expression_corrected(expression_data, sample_to_response, platform='Unknown'):
    """
    Perform differential expression analysis on log2-transformed data
    Using CORRECTED fold change calculation: mean_group1 - mean_group2
    """
    print(f"\nPerforming differential expression analysis for {platform}...")
    print("Using CORRECTED fold change calculation for log2 data")
    
    # Define comparisons
    comparisons = [
        ('EXT', 'NON', 'ext_vs_non'),
        ('EXT', 'INT', 'ext_vs_int'), 
        ('INT', 'NON', 'int_vs_non')
    ]
    
    results = {}
    
    for group1, group2, comp_name in comparisons:
        print(f"  Analyzing {group1} vs {group2}...")
        
        # Get samples for each group
        group1_samples = [s for s in expression_data.columns if sample_to_response.get(s) == group1]
        group2_samples = [s for s in expression_data.columns if sample_to_response.get(s) == group2]
        
        print(f"    {group1}: {len(group1_samples)} samples, {group2}: {len(group2_samples)} samples")
        
        if len(group1_samples) < 2 or len(group2_samples) < 2:
            print(f"    Insufficient samples for comparison: need at least 2 per group.")
            continue
        
        gene_results = {}
        
        for gene in expression_data.index:
            # Get expression values for each group
            # FIX: Use .values.ravel() to flatten the data, handling cases where duplicate
            # indices cause .loc to return a DataFrame instead of a Series.
            # Ensure columns are present before slicing
            
            group1_data = expression_data.loc[gene, [s for s in group1_samples if s in expression_data.columns]]
            group2_data = expression_data.loc[gene, [s for s in group2_samples if s in expression_data.columns]]


            group1_vals = pd.Series(pd.to_numeric(group1_data.values.ravel(), errors='coerce')).dropna()
            group2_vals = pd.Series(pd.to_numeric(group2_data.values.ravel(), errors='coerce')).dropna()
            
            # Skip if insufficient data after dropping NaNs
            if len(group1_vals) < 2 or len(group2_vals) < 2:
                continue
            
            # CORRECTED: For log2 transformed data, log2FC is the difference in means
            log2_fold_change = group1_vals.mean() - group2_vals.mean()
            
            # Calculate true fold change from log2FC
            fold_change = 2 ** log2_fold_change
            
            # Perform statistical test
            try:
                statistic, p_val = mannwhitneyu(group1_vals, group2_vals, alternative='two-sided')
            except ValueError as e: # Catch specific error for insufficient data for U-test
                # print(f"    Skipping gene {gene} due to Mann-Whitney U test error: {e}")
                continue
            except Exception as e:
                # print(f"    Skipping gene {gene} due to unexpected statistical test error: {e}")
                continue
            
            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(group1_vals), len(group2_vals)
            # Avoid division by zero if n1*n2 is zero
            effect_size = (2 * statistic / (n1 * n2) - 1) if (n1 * n2) > 0 else np.nan
            
            gene_results[gene] = {
                'log2_fold_change': log2_fold_change,
                'fold_change': fold_change,
                'p_value': p_val,
                'mean_group1': group1_vals.mean(),
                'mean_group2': group2_vals.mean(),
                'median_group1': group1_vals.median(),
                'median_group2': group2_vals.median(),
                'n_group1': len(group1_vals),
                'n_group2': len(group2_vals),
                'effect_size': effect_size,
                'statistic': statistic
            }
        
        results[comp_name] = gene_results
        print(f"    Analyzed {len(gene_results)} genes/proteins")
    
    return results

# ============================================================================
# GSEA ANALYSIS USING GSEAPY
# ============================================================================

def run_gsea_analysis_corrected(de_results, platform, output_dir):
    """Run GSEA analysis using gseapy with corrected rankings"""
    
    print(f"\nRunning GSEA analysis for {platform}...")
    
    # Gene sets to use
    gene_sets = [
        'GO_Biological_Process_2021',
        'GO_Molecular_Function_2021', 
        'KEGG_2021_Human',
        'Reactome_2022',
        'WikiPathway_2021_Human'
    ]
    
    all_gsea_results = {}
    
    for comp_name, results in de_results.items():
        print(f"\n  Processing {comp_name}...")
        
        if not results:
            print(f"    No differential expression results for {comp_name}. Skipping GSEA.")
            continue
        
        # Create ranked gene list using CORRECTED log2FC
        ranked_genes = []
        for gene, stats in results.items():
            # Ensure stats['log2_fold_change'] is a number
            if pd.notna(stats['log2_fold_change']):
                ranked_genes.append([gene, stats['log2_fold_change']])
        
        if not ranked_genes:
            print(f"    No valid genes for ranking in {comp_name}. Skipping GSEA for this comparison.")
            continue

        # Sort by log2 fold change (descending)
        ranked_genes.sort(key=lambda x: x[1], reverse=True)
        
        # Create DataFrame for gseapy
        ranked_df = pd.DataFrame(ranked_genes, columns=['gene', 'score'])
        
        print(f"    Ranked {len(ranked_df)} genes")
        print(f"    Score range: {ranked_df['score'].min():.3f} to {ranked_df['score'].max():.3f}")
        
        # Save ranked list
        ranked_file = f"{output_dir}/{platform}_{comp_name}_ranked_genes_corrected.rnk"
        ranked_df.to_csv(ranked_file, sep='\t', index=False, header=False)
        print(f"    Saved ranked list to: {ranked_file}")
        
        comp_gsea_results = {}
        
        # Run GSEA for each gene set
        for gene_set in gene_sets:
            print(f"      Running GSEA for {gene_set}...")
            
            try:
                # Run prerank GSEA
                pre_res = gp.prerank(
                    rnk=ranked_df,
                    gene_sets=gene_set,
                    processes=1, # Keep as 1 for better debugging in Jupyter
                    permutation_num=5000,
                    outdir=f"{output_dir}/{platform}_{comp_name}_{gene_set.replace(' ', '_')}_corrected",
                    format='png',
                    seed=42,
                    min_size=15,
                    max_size=500,
                    weighted_score_type=1
                )
                
                if hasattr(pre_res, 'res2d') and not pre_res.res2d.empty: # Check if dataframe is not empty
                    comp_gsea_results[gene_set] = pre_res.res2d
                    print(f"        Found {len(pre_res.res2d)} enriched pathways")
                else:
                    print(f"        No enriched pathways found for {gene_set}")
                    
            except Exception as e:
                print(f"        Error running GSEA for {gene_set}: {e}")
        
        all_gsea_results[comp_name] = comp_gsea_results
    
    return all_gsea_results

# ============================================================================
# CLASSIC PATHWAY PLOTS
# ============================================================================

def create_classic_pathway_plots(gsea_results, platform, output_dir):
    """Create enhanced classic GSEA pathway plots"""
    
    print(f"\nCreating enhanced classic pathway plots for {platform}...")
    
    for comp_name, comp_results in gsea_results.items():
        for gene_set, results_df in comp_results.items():
            if results_df is None or results_df.empty: # Use .empty for DataFrame check
                continue
            
            # Get significant pathways
            sig_pathways = results_df[results_df['FDR q-val'] < 0.25].head(20)
            
            if len(sig_pathways) == 0:
                print(f"  No significant pathways (FDR < 0.25) for {comp_name} - {gene_set}. Skipping plot.")
                continue
            
            print(f"  Creating plots for {comp_name} - {gene_set}: {len(sig_pathways)} pathways")
            
            # Create enhanced summary plot
            fig, ax = plt.subplots(figsize=(14, max(10, len(sig_pathways) * 0.4)))
            
            # Sort by NES for better visualization
            sig_pathways_sorted = sig_pathways.sort_values('NES')
            
            # Prepare data
            y_pos = np.arange(len(sig_pathways_sorted))
            
            # Enhanced color scheme based on significance and direction
            colors = []
            for _, row in sig_pathways_sorted.iterrows():
                if row['FDR q-val'] < 0.05:
                    color = '#b30000' if row['NES'] > 0 else '#003d82'  # Dark colors for very significant
                elif row['FDR q-val'] < 0.10:
                    color = '#d62728' if row['NES'] > 0 else '#1f77b4'  # Medium colors
                else:
                    color = '#ff7f7f' if row['NES'] > 0 else '#7fbfff'  # Light colors for less significant
                colors.append(color)
            
            # Create horizontal bar plot
            bars = ax.barh(y_pos, sig_pathways_sorted['NES'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_yticks(y_pos)
            
            # Enhanced pathway name handling
            pathway_labels = []
            for term in sig_pathways_sorted['Term']:
                term_str = str(term)
                # Clean up common pathway prefixes
                term_str = term_str.replace('GO:', '').replace('KEGG_', '').replace('WP_', '')
                
                if len(term_str) > 60:
                    term_str = term_str[:57] + "..."
                
                # Wrap for better display
                wrapped = '\n'.join(textwrap.wrap(term_str, width=45))
                pathway_labels.append(wrapped)
            
            ax.set_yticklabels(pathway_labels, fontsize=9)
            ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=14, fontweight='bold')
            ax.set_title(f'{platform} - {comp_name.replace("_", " ").upper()}\n{gene_set}\n(FDR < 0.25, Corrected Analysis)', 
                        fontsize=16, fontweight='bold')
            
            # Add reference lines
            ax.axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax.axvline(1.5, color='gray', linestyle='--', alpha=0.5, label='NES = ±1.5')
            ax.axvline(-1.5, color='gray', linestyle='--', alpha=0.5)
            
            # Add FDR annotations on the right
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(y_pos)
            
            fdr_labels = []
            for _, row in sig_pathways_sorted.iterrows():
                if row['FDR q-val'] < 0.001:
                    fdr_labels.append('***')
                elif row['FDR q-val'] < 0.01:
                    fdr_labels.append('**')
                elif row['FDR q-val'] < 0.05:
                    fdr_labels.append('*')
                else:
                    fdr_labels.append(f"{row['FDR q-val']:.3f}")
            
            ax2.set_yticklabels(fdr_labels, fontsize=8)
            ax2.set_ylabel('FDR q-value', fontsize=12, fontweight='bold')
            
            # Enhanced legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#b30000', alpha=0.8, label='Upregulated (FDR < 0.05)'),
                Patch(facecolor='#d62728', alpha=0.8, label='Upregulated (FDR < 0.10)'),
                Patch(facecolor='#ff7f7f', alpha=0.8, label='Upregulated (FDR < 0.25)'),
                Patch(facecolor='#003d82', alpha=0.8, label='Downregulated (FDR < 0.05)'),
                Patch(facecolor='#1f77b4', alpha=0.8, label='Downregulated (FDR < 0.10)'),
                Patch(facecolor='#7fbfff', alpha=0.8, label='Downregulated (FDR < 0.25)')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10, 
                     title='Significance Level', title_fontsize=11)
            
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            # Save plot
            safe_gene_set = gene_set.replace(' ', '_').replace('/', '_')
            plot_file = f"{output_dir}/plots/{platform}_{comp_name}_{safe_gene_set}_classic_enhanced.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"    Saved: {plot_file}")

def create_comprehensive_summary_corrected(nano_gsea, olink_gsea, nano_de, olink_de, output_dir):
    """Create comprehensive summary with correction indicators"""
    
    print("\nCreating comprehensive summary (corrected analysis)...")
    
    summary_data = []
    
    # Process Nanostring results
    for comp_name, comp_results in nano_gsea.items():
        for gene_set, results_df in comp_results.items():
            if results_df is not None and not results_df.empty: # Use .empty for DataFrame check
                sig_025 = len(results_df[results_df['FDR q-val'] < 0.25])
                sig_010 = len(results_df[results_df['FDR q-val'] < 0.10])
                sig_005 = len(results_df[results_df['FDR q-val'] < 0.05])
                
                summary_data.append({
                    'Platform': 'Nanostring',
                    'Comparison': comp_name.replace('_', ' ').upper(),
                    'Gene_Set': gene_set,
                    'Total_Pathways': len(results_df),
                    'Significant_FDR_0.25': sig_025,
                    'Significant_FDR_0.10': sig_010,
                    'Significant_FDR_0.05': sig_005,
                    'Top_Pathway': results_df.iloc[0]['Term'] if not results_df.empty else '',
                    'Top_NES': results_df.iloc[0]['NES'] if not results_df.empty else np.nan,
                    'Top_FDR': results_df.iloc[0]['FDR q-val'] if not results_df.empty else np.nan,
                    'Analysis_Version': 'Corrected_Log2FC'
                })
    
    # Process Olink results
    for comp_name, comp_results in olink_gsea.items():
        for gene_set, results_df in comp_results.items():
            if results_df is not None and not results_df.empty: # Use .empty for DataFrame check
                sig_025 = len(results_df[results_df['FDR q-val'] < 0.25])
                sig_010 = len(results_df[results_df['FDR q-val'] < 0.10])
                sig_005 = len(results_df[results_df['FDR q-val'] < 0.05])
                
                summary_data.append({
                    'Platform': 'Olink',
                    'Comparison': comp_name.replace('_', ' ').upper(),
                    'Gene_Set': gene_set,
                    'Total_Pathways': len(results_df),
                    'Significant_FDR_0.25': sig_025,
                    'Significant_FDR_0.10': sig_010,
                    'Significant_FDR_0.05': sig_005,
                    'Top_Pathway': results_df.iloc[0]['Term'] if not results_df.empty else '',
                    'Top_NES': results_df.iloc[0]['NES'] if not results_df.empty else np.nan,
                    'Top_FDR': results_df.iloc[0]['FDR q-val'] if not results_df.empty else np.nan,
                    'Analysis_Version': 'Corrected_Log2FC'
                })
    
    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/comprehensive_gsea_summary_corrected.csv", index=False)
        print(f"Saved corrected comprehensive summary: {output_dir}/comprehensive_gsea_summary_corrected.csv")
        
        # Create enhanced summary visualization
        create_summary_visualization_enhanced(summary_df, output_dir)
        
        return summary_df
    else:
        print("No summary data to save")
        return pd.DataFrame()

def create_summary_visualization_enhanced(summary_df, output_dir):
    """Create enhanced summary visualization"""
    
    if summary_df.empty:
        return
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    platforms = ['Nanostring', 'Olink']
    significance_levels = ['Significant_FDR_0.25', 'Significant_FDR_0.05']
    significance_labels = ['FDR < 0.25', 'FDR < 0.05']
    
    for p_idx, platform in enumerate(platforms):
        platform_data = summary_df[summary_df['Platform'] == platform]
        
        for s_idx, (sig_col, sig_label) in enumerate(zip(significance_levels, significance_labels)):
            ax = axes[s_idx, p_idx]
            
            if platform_data.empty:
                ax.text(0.5, 0.5, f'No {platform} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=16)
                ax.set_title(f'{platform} - {sig_label}', fontsize=16, fontweight='bold')
                continue
            
            # Group by comparison and sum significant pathways
            comp_summary = platform_data.groupby('Comparison')[sig_col].sum().sort_values(ascending=True)
            
            if len(comp_summary) > 0:
                bars = ax.barh(range(len(comp_summary)), comp_summary.values, 
                             color=['#d62728', '#1f77b4', '#2ca02c'][:len(comp_summary)], alpha=0.7)
                
                ax.set_yticks(range(len(comp_summary)))
                ax.set_yticklabels(comp_summary.index, fontsize=12)
                ax.set_xlabel(f'Number of Significant Pathways\n({sig_label})', fontsize=12)
                ax.set_title(f'{platform} - {sig_label}\n(Corrected Analysis)', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add value labels on bars
                for bar, value in zip(bars, comp_summary.values):
                    width = bar.get_width()
                    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                           f'{int(value)}', ha='left', va='center', fontweight='bold', fontsize=11)
                
                # Set appropriate x-limits
                ax.set_xlim(0, max(comp_summary.values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/comprehensive_summary_corrected.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved enhanced summary visualization: {output_dir}/plots/comprehensive_summary_corrected.png")

def save_tiered_gsea_results(gsea_results, platform, output_dir):
    """
    Saves GSEA results into separate, tiered files based on FDR thresholds.
    """
    print(f"\nSaving tiered GSEA results for {platform}...")

    for comp_name, comp_results in gsea_results.items():
        for gene_set, results_df in comp_results.items():
            if results_df is None or results_df.empty:
                continue

            safe_gene_set = gene_set.replace(' ', '_').replace('/', '_')
            base_filename = f"{output_dir}/{platform}_{comp_name}_{safe_gene_set}_gsea"

            # Define tiers
            tiers = {
                '0.05': results_df[results_df['FDR q-val'] < 0.05],
                '0.10': results_df[results_df['FDR q-val'] < 0.10],
                '0.25': results_df[results_df['FDR q-val'] < 0.25]
            }

            # Save a file for each tier if it contains any pathways
            for fdr, df_tier in tiers.items():
                if not df_tier.empty:
                    tier_filename = f"{base_filename}_fdr_{fdr}.csv"
                    df_tier.to_csv(tier_filename, index=False)
                    print(f"  Saved {len(df_tier)} pathways to: {tier_filename}")
                
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function with corrected analysis pipeline"""
    
    print("Starting CORRECTED GSEA analysis...")
    
    # Load data using verified approach
    sheets, sample_info_df, nanostring_df, olink_df = load_data()
    
    if sheets is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create sample mappings using verified approach
    sample_to_response, sample_to_patientid, sample_to_timepoint, mrn_to_response = create_sample_mappings(sheets, sample_info_df)
    
    if not sample_to_response:
        print("Failed to create sample mappings. Exiting.")
        return
    
    # Process Nanostring data
    nano_expression = process_nanostring_data_corrected(nanostring_df, sample_to_response)
    
    # Check if nano_expression is empty before proceeding
    if nano_expression.empty:
        print("Nanostring expression data is empty or invalid. Skipping Nanostring analysis.")
        nano_de_results = {}
        nano_gsea_results = {}
    else:
        # Perform CORRECTED differential expression analysis for Nanostring
        print("\n" + "="*60)
        print("CORRECTED DIFFERENTIAL EXPRESSION ANALYSIS - NANOSTRING")
        print("Using proper log2FC calculation: mean_group1 - mean_group2")
        print("="*60)
        nano_de_results = perform_differential_expression_corrected(nano_expression, sample_to_response, 'Nanostring')

        # Save differential expression results for Nanostring
        for comp_name, results in nano_de_results.items():
            if results:
                de_df = pd.DataFrame.from_dict(results, orient='index')
                de_df.index.name = 'Gene'
                de_df.to_csv(f"gsea_results_corrected/nanostring/nanostring_{comp_name}_differential_expression_corrected.csv")

        # Run CORRECTED GSEA analysis for Nanostring
        print("\n" + "="*60)
        print("CORRECTED GSEA ANALYSIS - NANOSTRING")
        print("="*60)
        nano_gsea_results = run_gsea_analysis_corrected(nano_de_results, 'Nanostring', 'gsea_results_corrected/nanostring')
    
    # Process Olink data
    olink_expression, olink_sample_to_response = process_olink_data_corrected(olink_df, sample_info_df, sample_to_response, mrn_to_response)
    
    # Check if olink_expression is empty before proceeding
    if olink_expression is None or olink_expression.empty:
        print("Olink expression data is empty or invalid. Skipping Olink analysis.")
        olink_de_results = {}
        olink_gsea_results = {}
    else:
        # Perform CORRECTED differential expression analysis for Olink
        print("\n" + "="*60)
        print("CORRECTED DIFFERENTIAL EXPRESSION ANALYSIS - OLINK")
        print("Using proper log2FC calculation: mean_group1 - mean_group2")
        print("="*60)
        olink_de_results = perform_differential_expression_corrected(olink_expression, olink_sample_to_response, 'Olink')
        
        # Save differential expression results for Olink
        for comp_name, results in olink_de_results.items():
            if results:
                de_df = pd.DataFrame.from_dict(results, orient='index')
                de_df.index.name = 'Protein'
                de_df.to_csv(f"gsea_results_corrected/olink/olink_{comp_name}_differential_expression_corrected.csv")
        
        # Run CORRECTED GSEA analysis for Olink
        print("\n" + "="*60)
        print("CORRECTED GSEA ANALYSIS - OLINK")
        print("="*60)
        olink_gsea_results = run_gsea_analysis_corrected(olink_de_results, 'Olink', 'gsea_results_corrected/olink')
    
    # Save GSEA results using the new tiered approach
    save_tiered_gsea_results(nano_gsea_results, 'nanostring', 'gsea_results_corrected/nanostring')
    save_tiered_gsea_results(olink_gsea_results, 'olink', 'gsea_results_corrected/olink')
    
    # Create enhanced classic pathway plots
    print("\n" + "="*60)
    print("CREATING ENHANCED CLASSIC PATHWAY PLOTS")
    print("="*60)
    
    create_classic_pathway_plots(nano_gsea_results, 'Nanostring', 'gsea_results_corrected')
    create_classic_pathway_plots(olink_gsea_results, 'Olink', 'gsea_results_corrected')
    
    # Create comprehensive summary
    comprehensive_summary = create_comprehensive_summary_corrected(
        nano_gsea_results, olink_gsea_results, 
        nano_de_results, olink_de_results, 
        'gsea_results_corrected'
    )
    
    # Final summary
    print("\n" + "="*80)
    print("CORRECTED ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    
    total_nano_genes = sum(len(results) for results in nano_de_results.values())
    total_olink_proteins = sum(len(results) for results in olink_de_results.values())
    
    print(f"Nanostring: {total_nano_genes} total gene comparisons")
    print(f"Olink: {total_olink_proteins} total protein comparisons")
    
    if not comprehensive_summary.empty:
        nano_pathways = comprehensive_summary[comprehensive_summary['Platform'] == 'Nanostring']['Significant_FDR_0.25'].sum()
        olink_pathways = comprehensive_summary[comprehensive_summary['Platform'] == 'Olink']['Significant_FDR_0.25'].sum()
        print(f"Nanostring: {nano_pathways} significant pathways (FDR < 0.25)")
        print(f"Olink: {olink_pathways} significant pathways (FDR < 0.25)")
    
    print(f"\nKey corrections implemented:")
    print(f"✓ Log2FC calculation: mean_group1 - mean_group2 (not log2(mean_group1/mean_group2))")
    print(f"✓ Proper Excel file structure handling")
    print(f"✓ Verified sample mapping approach")
    print(f"✓ Olink data restructuring with control exclusion")
    print(f"✓ Enhanced pathway visualizations")
    print(f"✓ Tiered GSEA results saved to separate CSV files (FDR < 0.05, 0.10, 0.25)")
    print(f"\nAll results saved in 'gsea_results_corrected/' directory")

# Call the main function to start the analysis
if __name__ == "__main__":
    main()


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import textwrap

# --- Re-using plotting style from your original script ---
plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
# --- End of re-used plotting style ---


def generate_ext_vs_non_pathway_summary_figure_refined(output_dir='gsea_results_corrected'):
    """
    Generates a refined figure summarizing the number of enriched pathways
    for the 'EXT vs NON' comparison across platforms and FDR thresholds,
    with enrichment direction shown by y-axis position.
    """
    
    print("\n" + "="*80)
    print("GENERATING REFINED EXT vs NON PATHWAY SUMMARY FIGURE (Bidirectional Y-axis)")
    print("="*80)

    platforms = ['nanostring', 'olink']
    gene_sets = [
        'GO_Biological_Process_2021',
        'GO_Molecular_Function_2021',
        'KEGG_2021_Human',
        'Reactome_2022', 
        'WikiPathway_2021_Human'
    ] 
    fdr_thresholds = ['0.05', '0.10', '0.25']
    comparison_name = 'ext_vs_non'

    summary_data = []

    for platform in platforms:
        platform_output_dir = os.path.join(output_dir, platform)

        for gene_set in gene_sets:
            safe_gene_set_name = gene_set.replace(' ', '_').replace('/', '_')
            base_pattern = f"{platform_output_dir}/{platform}_{comparison_name}_{safe_gene_set_name}_gsea_fdr_"

            for fdr in fdr_thresholds:
                filename = f"{base_pattern}{fdr}.csv"
                
                up_count = 0
                down_count = 0

                if os.path.exists(filename):
                    try:
                        df = pd.read_csv(filename)
                        if not df.empty:
                            up_count = len(df[df['NES'] > 0])
                            down_count = len(df[df['NES'] < 0])
                            
                        if up_count > 0:
                            summary_data.append({
                                'Platform': platform.capitalize(),
                                'FDR_Threshold': fdr,
                                'Direction': 'Enriched (Upregulated in EXT)', 
                                'Count': up_count, # Positive count for plotting above 0
                                'Gene_Set': gene_set 
                            })
                        if down_count > 0:
                            summary_data.append({
                                'Platform': platform.capitalize(),
                                'FDR_Threshold': fdr,
                                'Direction': 'Enriched (Downregulated in EXT)', 
                                'Count': -down_count, # Negative count for plotting below 0
                                'Gene_Set': gene_set 
                            })

                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

    if not summary_data:
        print("No significant pathways found for EXT vs NON comparison to plot.")
        return

    summary_df = pd.DataFrame(summary_data)
    
    # Aggregate counts by Platform, FDR, and Direction
    aggregated_df = summary_df.groupby(['Platform', 'FDR_Threshold', 'Direction'])['Count'].sum().reset_index()

    # Define the order for plotting
    fdr_order = ['0.05', '0.10', '0.25']
    aggregated_df['FDR_Threshold'] = pd.Categorical(aggregated_df['FDR_Threshold'], categories=fdr_order, ordered=True)
    aggregated_df = aggregated_df.sort_values(['Platform', 'FDR_Threshold', 'Direction'])

    # Determine consistent y-limits for both positive and negative values
    max_abs_count = aggregated_df['Count'].abs().max()
    consistent_ylim = max_abs_count * 1.2 # Give some buffer
    
    # --- Plotting ---
    g = sns.catplot(
        data=aggregated_df,
        x='FDR_Threshold',
        y='Count',
        hue='Direction',
        col='Platform', 
        kind='bar', 
        palette={'Enriched (Upregulated in EXT)': '#377eb8', 'Enriched (Downregulated in EXT)': '#e41a1c'}, # R&B color scheme
        edgecolor='black',
        linewidth=1,
        errorbar=None, 
        height=5, aspect=1.3, 
        col_wrap=2, 
        legend_out=True 
    )

    g.fig.suptitle(f'GSEA Pathway Enrichment (EXT vs NON Comparison)\nAcross Platforms and FDR Thresholds', 
                 fontsize=16, y=1.05, ha='center', fontweight='bold')

    for ax in g.axes.flat:
        current_platform = ax.get_title().split(' = ')[-1]
        ax.set_title(f'{current_platform} Data', fontsize=14, fontweight='bold', pad=10)
        
        ax.set_xlabel('FDR Threshold', fontsize=12)
        ax.set_ylabel('Number of Pathways', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='y', alpha=0.75, linestyle='--')
        ax.set_ylim(-consistent_ylim, consistent_ylim) # Apply consistent bidirectional Y-limit
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8) # Add a line at y=0

        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height != 0: # Only label if bar has height
                    # Annotate with absolute value and adjust position based on positive/negative
                    annotation_text = f'{int(abs(height))}'
                    y_offset = 3 if height > 0 else -15 # Adjust offset for negative bars
                    ax.annotate(annotation_text,
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, y_offset), 
                                textcoords="offset points",
                                ha='center', va='bottom' if height > 0 else 'top', 
                                fontsize=9, fontweight='bold', color='black')
    
    handles, labels = g.axes.flat[0].get_legend_handles_labels() 
    g.legend.set_title('Enrichment Direction')
    for text in g.legend.get_texts():
        text.set_fontsize(10) 
    g.legend.get_title().set_fontsize(11)

    g.tight_layout(rect=[0, 0, 1, 0.96])
    
    figure_path = os.path.join(output_dir, 'plots', 'ext_vs_non_pathway_summary_figure_bidirectional.png')
    plt.savefig(figure_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Generated refined pathway summary figure: {figure_path}")


# --- How to run this snippet ---
# Call the function to generate the figure:
generate_ext_vs_non_pathway_summary_figure_refined(output_dir='gsea_results_corrected')


# In[ ]:




