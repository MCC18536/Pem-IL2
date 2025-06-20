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
print("PHASE 1 COMPLETE")
print("="*60)

# Store for Phase 2
phase1_data = {
    'sheets': sheets,
    'sample_to_response': sample_to_response,
    'sample_to_patientid': sample_to_patientid,
    'sample_to_timepoint': sample_to_timepoint
}


# In[3]:


#!/usr/bin/env python
# coding: utf-8

"""
Phase 2: Exploratory Data Analysis Module
Run this after Phase 1 data verification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'

class Phase2Explorer:
    """Exploratory Data Analysis for multi-omics biomarker study"""
    
    def __init__(self, phase1_data):
        """Initialize with Phase 1 results"""
        self.sheets = phase1_data['sheets']
        self.sample_to_response = phase1_data['sample_to_response']
        self.sample_to_patientid = phase1_data['sample_to_patientid']
        self.sample_to_timepoint = phase1_data['sample_to_timepoint']
        
        # Load data
        self.nanostring_df = self.sheets['nanostring']
        self.olink_df = self.sheets['Olink']
        self.sample_info_df = self.sheets['sampleID']
        
        print("Phase 2 Explorer initialized")
        print(f"Available samples: {len(self.sample_to_response)}")
        
    def create_cohort_summary(self):
        """Step 2.1: Cohort Characterization"""
        print("STEP 2.1: COHORT CHARACTERIZATION")
        print("="*50)
        
        # Patient distribution by response group
        patient_responses = {}
        for sample, response in self.sample_to_response.items():
            if sample in self.sample_to_patientid:
                patient_id = self.sample_to_patientid[sample]
                patient_responses[patient_id] = response
        
        response_counts = {}
        for response in patient_responses.values():
            response_counts[response] = response_counts.get(response, 0) + 1
        
        print("\nPatient Distribution by Response Group:")
        for response, count in sorted(response_counts.items()):
            print(f"  {response}: {count} patients")
        
        # Timepoint availability analysis
        timepoint_data = {}
        for sample, timepoint_info in self.sample_to_timepoint.items():
            if sample in self.sample_to_patientid:
                patient_id = self.sample_to_patientid[sample]
                response = self.sample_to_response[sample]
                timepoint = timepoint_info['timepoint']
                
                key = (patient_id, response)
                if key not in timepoint_data:
                    timepoint_data[key] = []
                timepoint_data[key].append(timepoint)
        
        # Create timepoint availability summary
        all_timepoints = ['B1W1', 'B2W2', 'B2W4', 'B4W1']
        timepoint_summary = []
        
        for (patient_id, response), timepoints in timepoint_data.items():
            row = {'Patient_ID': patient_id, 'Response': response}
            for tp in all_timepoints:
                row[tp] = 'Yes' if tp in timepoints else 'No'
            row['Total_Timepoints'] = len(timepoints)
            timepoint_summary.append(row)
        
        timepoint_df = pd.DataFrame(timepoint_summary)
        
        print(f"\nTimepoint Availability Summary:")
        print(f"Total patients with data: {len(timepoint_df)}")
        
        for response in sorted(timepoint_df['Response'].unique()):
            subset = timepoint_df[timepoint_df['Response'] == response]
            print(f"\n{response} group ({len(subset)} patients):")
            for tp in all_timepoints:
                available = (subset[tp] == 'Yes').sum()
                print(f"  {tp}: {available}/{len(subset)} ({100*available/len(subset):.1f}%)")
        
        # Patients with all timepoints
        complete_patients = timepoint_df[timepoint_df['Total_Timepoints'] == 4]
        print(f"\nPatients with all 4 timepoints: {len(complete_patients)}")
        for response in sorted(complete_patients['Response'].unique()):
            count = (complete_patients['Response'] == response).sum()
            print(f"  {response}: {count}")
        
        return timepoint_df
    
    def create_data_structure_plots(self):
        """Step 2.2: Overall Data Structure Visualization"""
        print("\nSTEP 2.2: DATA STRUCTURE VISUALIZATION")
        print("="*50)
        
        # Prepare nanostring data for PCA
        expr_cols = [col for col in self.nanostring_df.columns if col in self.sample_to_response]
        
        if len(expr_cols) < 5:
            print("Warning: Insufficient expression columns for PCA analysis")
            return None
        
        # Create expression matrix (genes x samples)
        gene_col = self.nanostring_df.columns[0]
        genes = self.nanostring_df[gene_col].dropna().tolist()
        
        expr_matrix = []
        sample_labels = []
        response_labels = []
        timepoint_labels = []
        
        for sample in expr_cols:
            if sample in self.sample_to_response:
                sample_data = []
                for gene in genes:
                    gene_rows = self.nanostring_df[self.nanostring_df[gene_col] == gene]
                    if len(gene_rows) > 0:
                        val = pd.to_numeric(gene_rows[sample].iloc[0], errors='coerce')
                        sample_data.append(val if pd.notna(val) else 0)
                    else:
                        sample_data.append(0)
                
                if len(sample_data) == len(genes):
                    expr_matrix.append(sample_data)
                    sample_labels.append(sample)
                    response_labels.append(self.sample_to_response[sample])
                    
                    # Get timepoint
                    if sample in self.sample_to_timepoint:
                        timepoint_labels.append(self.sample_to_timepoint[sample]['timepoint'])
                    else:
                        timepoint_labels.append('Unknown')
        
        if len(expr_matrix) == 0:
            print("Error: No valid expression data found")
            return None
        
        expr_matrix = np.array(expr_matrix).T  # Transpose to genes x samples
        
        print(f"Expression matrix shape: {expr_matrix.shape}")
        print(f"Samples: {len(sample_labels)}")
        print(f"Genes: {len(genes)}")
        
        # Remove genes with too many missing values
        valid_genes_mask = np.mean(expr_matrix == 0, axis=1) < 0.5
        expr_matrix_clean = expr_matrix[valid_genes_mask]
        genes_clean = [genes[i] for i in range(len(genes)) if valid_genes_mask[i]]
        
        print(f"After filtering: {expr_matrix_clean.shape[0]} genes")
        
        # PCA Analysis
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_matrix_clean.T)  # Scale samples
        
        pca = PCA(n_components=min(10, expr_scaled.shape[0]-1))
        pca_result = pca.fit_transform(expr_scaled)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # PCA by response group
        ax1 = plt.subplot(2, 4, 1)
        colors = {'EXT': 'red', 'INT': 'blue', 'NON': 'gray'}
        for response in colors:
            mask = np.array(response_labels) == response
            if np.sum(mask) > 0:
                plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                           c=colors[response], label=f'{response} (n={np.sum(mask)})',
                           s=60, alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('PCA by Response Group')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # PCA by timepoint
        ax2 = plt.subplot(2, 4, 2)
        timepoint_colors = {'B1W1': 'purple', 'B2W2': 'green', 'B2W4': 'orange', 'B4W1': 'brown'}
        for timepoint in timepoint_colors:
            mask = np.array(timepoint_labels) == timepoint
            if np.sum(mask) > 0:
                plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                           c=timepoint_colors[timepoint], label=f'{timepoint} (n={np.sum(mask)})',
                           s=60, alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('PCA by Timepoint')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Scree plot
        ax3 = plt.subplot(2, 4, 3)
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.grid(alpha=0.3)
        
        # Sample correlation heatmap
        ax4 = plt.subplot(2, 4, 4)
        if len(sample_labels) <= 50:  # Only if manageable number of samples
            sample_corr = np.corrcoef(expr_matrix_clean.T)
            im = plt.imshow(sample_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.title(f'Sample Correlation\n(n={len(sample_labels)})')
            plt.colorbar(im, shrink=0.8)
        else:
            plt.text(0.5, 0.5, f'Too many samples\nfor correlation plot\n(n={len(sample_labels)})', 
                    ha='center', va='center', transform=ax4.transAxes)
            plt.title('Sample Correlation')
        
        # Sample clustering dendrogram
        ax5 = plt.subplot(2, 1, 2)
        if len(sample_labels) <= 50:
            linkage_matrix = linkage(expr_scaled, method='ward')
            dend = dendrogram(linkage_matrix, labels=sample_labels, 
                            leaf_rotation=90, leaf_font_size=8)
            plt.title('Sample Clustering Dendrogram')
            plt.ylabel('Distance')
        else:
            plt.text(0.5, 0.5, f'Too many samples for dendrogram\n(n={len(sample_labels)})', 
                    ha='center', va='center', transform=ax5.transAxes)
            plt.title('Sample Clustering')
        
        plt.tight_layout()
        plt.savefig('phase2_data_structure.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary statistics
        print(f"\nPCA Summary:")
        print(f"  PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
        print(f"  PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
        print(f"  First 3 PCs explain {pca.explained_variance_ratio_[:3].sum():.1%} of variance")
        
        return {
            'pca_result': pca_result,
            'pca_model': pca,
            'expr_matrix': expr_matrix_clean,
            'sample_labels': sample_labels,
            'response_labels': response_labels,
            'timepoint_labels': timepoint_labels,
            'genes': genes_clean
        }
    
    def analyze_missing_data(self):
        """Analyze missing data patterns"""
        print("\nMISSING DATA ANALYSIS:")
        print("="*30)
        
        # Analyze sample availability across timepoints
        all_timepoints = ['B1W1', 'B2W2', 'B2W4', 'B4W1']
        missing_pattern = {}
        
        for patient_id in set(self.sample_to_patientid.values()):
            patient_samples = [s for s, p in self.sample_to_patientid.items() if p == patient_id]
            patient_timepoints = []
            
            for sample in patient_samples:
                if sample in self.sample_to_timepoint:
                    patient_timepoints.append(self.sample_to_timepoint[sample]['timepoint'])
            
            pattern = tuple(sorted(set(patient_timepoints)))
            missing_pattern[pattern] = missing_pattern.get(pattern, 0) + 1
        
        print("Timepoint patterns:")
        for pattern, count in sorted(missing_pattern.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count} patients")
        
        return missing_pattern
    
    def run_full_analysis(self):
        """Run complete Phase 2 analysis"""
        print("RUNNING PHASE 2: EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Step 2.1: Cohort characterization
        timepoint_df = self.create_cohort_summary()
        
        # Step 2.2: Data structure plots
        pca_results = self.create_data_structure_plots()
        
        # Missing data analysis
        missing_patterns = self.analyze_missing_data()
        
        print("\n" + "="*60)
        print("PHASE 2 COMPLETE")
        print("="*60)
        print("✓ Cohort characterization completed")
        print("✓ PCA and clustering analysis completed")
        print("✓ Missing data patterns identified")
        print("\nReady for Phase 3: Temporal Analysis")
        
        return {
            'timepoint_summary': timepoint_df,
            'pca_analysis': pca_results,
            'missing_patterns': missing_patterns
        }

phase2 = Phase2Explorer(phase1_data)
phase2_results = phase2.run_full_analysis()


# In[12]:


#!/usr/bin/env python
# coding: utf-8

"""
Response Group Comparison Analysis at Individual Timepoints
Compare gene expression between EXT, INT, and NON responders at each timepoint
Using p-value < 0.05 threshold and excluding control genes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
import re
warnings.filterwarnings('ignore')

# Set up visualization style
plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

class ResponseGroupTimePointAnalyzer:
    """Analyze gene expression differences between response groups at specific timepoints"""
    
    def __init__(self, phase1_data):
        """Initialize with Phase 1 data"""
        self.sheets = phase1_data['sheets']
        self.sample_to_response = phase1_data['sample_to_response']
        self.sample_to_patientid = phase1_data['sample_to_patientid']
        self.sample_to_timepoint = phase1_data['sample_to_timepoint']
        
        # Response group colors for consistent plotting
        self.group_colors = {
            'EXT': '#d62728',  # Red
            'INT': '#1f77b4',  # Blue  
            'NON': '#7f7f7f'   # Gray
        }
        
        self._prepare_expression_data()
        print("Response Group Timepoint Analyzer initialized")
    
    def _prepare_expression_data(self):
        """Prepare expression data from Phase 1, excluding control genes"""
        nanostring_df = self.sheets['nanostring']
        expr_cols = [col for col in nanostring_df.columns if col in self.sample_to_response]
        
        gene_col = nanostring_df.columns[0]
        all_genes = nanostring_df[gene_col].dropna().tolist()
        
        # Filter out control genes (NEG_*, POS_*, etc.)
        control_pattern = re.compile(r'^(NEG|POS)_[A-Z]')
        self.genes = [gene for gene in all_genes if not control_pattern.match(gene)]
        
        print(f"Filtered out {len(all_genes) - len(self.genes)} control genes")
        
        expr_matrix = []
        self.sample_labels = []
        
        for sample in expr_cols:
            if sample in self.sample_to_response:
                sample_data = []
                for gene in self.genes:
                    gene_rows = nanostring_df[nanostring_df[gene_col] == gene]
                    if len(gene_rows) > 0:
                        val = pd.to_numeric(gene_rows[sample].iloc[0], errors='coerce')
                        sample_data.append(val if pd.notna(val) else np.nan)
                    else:
                        sample_data.append(np.nan)
                
                expr_matrix.append(sample_data)
                self.sample_labels.append(sample)
        
        self.expr_matrix = np.array(expr_matrix).T  # genes x samples
        print(f"Expression data prepared: {len(self.genes)} genes, {len(self.sample_labels)} samples")
    
    def analyze_timepoint(self, timepoint, p_threshold=0.05, fc_threshold=0.5):
        """Analyze expression differences at a specific timepoint"""
        print(f"\nANALYZING TIMEPOINT: {timepoint}")
        print("="*50)
        
        # Get samples for this timepoint
        tp_samples = [s for s in self.sample_labels 
                     if s in self.sample_to_timepoint and 
                     self.sample_to_timepoint[s]['timepoint'] == timepoint]
        
        # Group samples by response category
        group_samples = {}
        for group in ['EXT', 'INT', 'NON']:
            group_samples[group] = [s for s in tp_samples 
                                  if self.sample_to_response.get(s) == group]
            print(f"  {group} group: {len(group_samples[group])} samples")
        
        # Perform pairwise comparisons
        comparisons = [
            ('EXT', 'NON', 'EXT vs NON'),
            ('INT', 'NON', 'INT vs NON'),
            ('EXT', 'INT', 'EXT vs INT')
        ]
        
        comparison_results = {}
        
        for group1, group2, comparison_name in comparisons:
            print(f"\n  COMPARING {group1} vs {group2} at {timepoint}")
            
            group1_samples = group_samples[group1]
            group2_samples = group_samples[group2]
            
            # Skip comparison if either group has too few samples
            if len(group1_samples) < 2 or len(group2_samples) < 2:
                print(f"    ⚠️ Insufficient samples for {group1} vs {group2} comparison")
                continue
            
            # Perform gene-by-gene statistical test
            gene_results = []
            
            for gene_idx, gene in enumerate(self.genes):
                group1_expr = []
                group2_expr = []
                
                for sample in group1_samples:
                    idx = self.sample_labels.index(sample)
                    val = self.expr_matrix[gene_idx, idx]
                    if pd.notna(val):
                        group1_expr.append(val)
                
                for sample in group2_samples:
                    idx = self.sample_labels.index(sample)
                    val = self.expr_matrix[gene_idx, idx]
                    if pd.notna(val):
                        group2_expr.append(val)
                
                # Skip genes with insufficient data
                if len(group1_expr) < 2 or len(group2_expr) < 2:
                    continue
                
                # Calculate statistics
                try:
                    # Try t-test first
                    t_stat, p_val = stats.ttest_ind(group1_expr, group2_expr, equal_var=False)
                    
                    # Calculate effect size (log2 fold change of means)
                    mean1 = np.mean(group1_expr)
                    mean2 = np.mean(group2_expr)
                    log2fc = mean1 - mean2  # For log2-transformed data
                    abs_log2fc = abs(log2fc)
                    
                    # Calculate fold change for interpretability
                    fold_change = 2**abs_log2fc
                    
                    gene_results.append({
                        'gene': gene,
                        'p_value': p_val,
                        'log2fc': log2fc,
                        'abs_log2fc': abs_log2fc,
                        'fold_change': fold_change,
                        'mean_expr1': mean1,
                        'mean_expr2': mean2,
                        'n_samples1': len(group1_expr),
                        'n_samples2': len(group2_expr)
                    })
                except Exception as e:
                    # Skip gene if test fails
                    continue
            
            if len(gene_results) == 0:
                print(f"    ⚠️ No valid results for {group1} vs {group2} comparison")
                continue
            
            # Create results dataframe
            results_df = pd.DataFrame(gene_results)
            
            # Apply FDR correction
            _, fdr_p, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
            results_df['fdr_p'] = fdr_p
            
            # Count significant genes by different criteria
            p_sig = results_df[results_df['p_value'] < p_threshold]
            fdr_sig = results_df[results_df['fdr_p'] < p_threshold]
            
            # Count genes with substantial fold change
            fc_sig = results_df[results_df['abs_log2fc'] > fc_threshold]
            
            # Combined criteria (both p-value and fold change)
            p_fc_sig = results_df[(results_df['p_value'] < p_threshold) & 
                                (results_df['abs_log2fc'] > fc_threshold)]
            
            # Count up/down regulated genes for p-test + FC
            p_fc_up = p_fc_sig[p_fc_sig['log2fc'] > 0]
            p_fc_down = p_fc_sig[p_fc_sig['log2fc'] < 0]
            
            print(f"    Total genes tested: {len(results_df)}")
            print(f"    p < {p_threshold}: {len(p_sig)} genes")
            print(f"    FDR < {p_threshold}: {len(fdr_sig)} genes")
            print(f"    |log2FC| > {fc_threshold}: {len(fc_sig)} genes")
            print(f"    p < {p_threshold} AND |log2FC| > {fc_threshold}: {len(p_fc_sig)} genes")
            print(f"      Up in {group1}: {len(p_fc_up)}, Down in {group1}: {len(p_fc_down)}")
            
            # Show top genes by p-value
            if len(p_sig) > 0:
                top_sig = p_sig.nsmallest(10, 'p_value')
                
                print(f"\n    Top significant genes (by p-value):")
                for _, row in top_sig.iterrows():
                    direction = "UP" if row['log2fc'] > 0 else "DOWN"
                    print(f"      {row['gene']}: {direction} log2FC={row['log2fc']:.2f} ({row['fold_change']:.1f}-fold), p={row['p_value']:.4f}")
            
            comparison_results[comparison_name] = results_df
        
        return comparison_results
    
    def create_volcano_plot(self, results_df, title, figsize=(12, 9), 
                      p_threshold=0.05, fc_threshold=0.6, 
                      highlight_genes=None):
        """Create volcano plot with publication-quality formatting, matching the provided style"""
        if results_df is None or len(results_df) == 0:
            print(f"⚠️ No data for volcano plot: {title}")
            return None

        # Remove any rows with missing data
        results_df = results_df.dropna(subset=['log2fc', 'p_value'])

        # Create figure
        plt.figure(figsize=figsize)

        # Define thresholds
        P_THRESHOLD = p_threshold
        FC_THRESHOLD = fc_threshold

        print(f"Thresholds: Log2FC ±{FC_THRESHOLD}, p-value {P_THRESHOLD}")

        # Create significance categories
        results_df['meets_fc_threshold'] = abs(results_df['log2fc']) >= FC_THRESHOLD
        results_df['meets_p_threshold'] = results_df['p_value'] < P_THRESHOLD
        results_df['meets_both_criteria'] = results_df['meets_fc_threshold'] & results_df['meets_p_threshold']

        sig_both = results_df[results_df['meets_both_criteria']]
        sig_p_only = results_df[results_df['meets_p_threshold'] & ~results_df['meets_fc_threshold']]

        print(f"Total genes: {len(results_df)}")
        print(f"Meeting p<{P_THRESHOLD} only: {len(sig_p_only)}")
        print(f"Meeting both criteria: {len(sig_both)}")

        # Plot all points (base layer)
        plt.scatter(
            results_df['log2fc'],
            -np.log10(results_df['p_value']),
            alpha=0.6,
            s=40,
            color='#cccccc',
            edgecolor='none',
            zorder=1,
            label=f'Non-significant (n={len(results_df) - len(sig_p_only) - len(sig_both)})'
        )

        # Plot significant points meeting both criteria (red)
        if len(sig_both) > 0:
            plt.scatter(
                sig_both['log2fc'],
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
        y_max = max(-np.log10(results_df['p_value']))
        plt.text(0, -np.log10(P_THRESHOLD) + 0.05, f'p = {P_THRESHOLD}', 
                 color='#e41a1c', fontsize=10, ha='center', va='bottom')

        # LABEL SIGNIFICANT GENES (ONLY those meeting both criteria)
        genes_to_label = []

        # Label genes meeting both criteria (highest priority)
        if len(sig_both) > 0:
            genes_to_label.extend(sig_both.nsmallest(min(20, len(sig_both)), 'p_value').iterrows())

        # Add gene labels with smart positioning
        texts = []
        for _, row in genes_to_label:
            x = row['log2fc']
            y = -np.log10(row['p_value'])
            gene_name = row['gene']

            # Bold text for genes meeting both criteria
            text_color = 'black'
            font_weight = 'bold'

            # Add text with background box for visibility
            text = plt.text(x, y, gene_name, fontsize=9, fontweight=font_weight,
                           ha='center', va='bottom', color=text_color, zorder=4)
            texts.append(text)

        # Try to use adjustText if available for better label positioning
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
        plt.xlabel(f'Log2 Fold Change', fontsize=14, fontweight='bold')
        plt.ylabel('-Log10(p-value)', fontsize=14, fontweight='bold')

        plt.grid(alpha=0.3, linestyle='--', zorder=0)

        # Set axis limits with padding for labels
        x_range = max(abs(results_df['log2fc'].min()), abs(results_df['log2fc'].max()))
        plt.xlim(-x_range * 1.4, x_range * 1.4)
        plt.ylim(-0.1, y_max * 1.3)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        return plt.gcf(), sig_both
    
    def create_heatmap(self, timepoint, comparison_results, 
                 max_genes=50, p_threshold=0.05):
        """Create enhanced heatmap with colorbar on right, no separate response group bar, and RdBu_r colormap"""
        print(f"\nCreating heatmap for {timepoint}")

        if not comparison_results or len(comparison_results) == 0:
            print(f"⚠️ No comparison results for {timepoint}")
            return None

        # Get samples for this timepoint
        tp_samples = [s for s in self.sample_labels 
                     if s in self.sample_to_timepoint and 
                     self.sample_to_timepoint[s]['timepoint'] == timepoint]

        # Collect genes from all comparisons
        selected_genes = set()

        # Collect by p-value
        for comp_name, results_df in comparison_results.items():
            mask = (results_df['p_value'] < p_threshold)
            comp_genes = results_df.loc[mask, 'gene'].tolist()
            selected_genes.update(comp_genes)

        selected_genes = list(selected_genes)
        print(f"  Significant genes (p < {p_threshold}) across all comparisons: {len(selected_genes)}")

        # If no genes meet criteria, use top genes by p-value
        if len(selected_genes) == 0:
            print(f"  ⚠️ No genes meet p < {p_threshold}, using top genes by p-value")

            # Collect all genes and their best p-value
            all_genes = {}
            for comp_name, results_df in comparison_results.items():
                for _, row in results_df.iterrows():
                    gene = row['gene']
                    p_val = row['p_value']
                    if gene not in all_genes or p_val < all_genes[gene]:
                        all_genes[gene] = p_val

            # Sort genes by p-value and take top
            sorted_genes = sorted(all_genes.items(), key=lambda x: x[1])
            selected_genes = [gene for gene, p_val in sorted_genes[:max_genes]]

            print(f"  Using top {len(selected_genes)} genes by p-value")

        # Limit to max_genes if needed
        if len(selected_genes) > max_genes:
            # Collect gene scores (best p-values)
            gene_scores = {}
            for comp_name, results_df in comparison_results.items():
                for _, row in results_df.iterrows():
                    gene = row['gene']
                    if gene in selected_genes:
                        p_val = row['p_value']
                        if gene not in gene_scores or p_val < gene_scores[gene]:
                            gene_scores[gene] = p_val

            # Sort genes by p-value and take top max_genes
            sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1])
            selected_genes = [gene for gene, score in sorted_genes[:max_genes]]

            print(f"  ⚠️ Limiting heatmap to top {max_genes} genes by p-value")

        # Prepare heatmap data
        gene_indices = []

        for gene in selected_genes:
            if gene in self.genes:
                gene_idx = self.genes.index(gene)
                gene_indices.append(gene_idx)

        sample_indices = []
        sample_groups = []

        for sample in tp_samples:
            if sample in self.sample_to_response:
                sample_idx = self.sample_labels.index(sample)
                sample_indices.append(sample_idx)
                sample_groups.append(self.sample_to_response[sample])

        # Extract expression data
        expr_data = self.expr_matrix[gene_indices, :][:, sample_indices]

        # Create sample labels
        sample_labels = []
        for i, sample_idx in enumerate(sample_indices):
            sample = self.sample_labels[sample_idx]
            group = sample_groups[i]
            # Get patient ID if available
            patient_id = self.sample_to_patientid.get(sample, 'Unknown')
            # Format: "016 (EXT)" - more readable
            patient_short = patient_id.split('-')[-1] if '-' in patient_id else patient_id
            sample_labels.append(f"{patient_short} ({group})")

        # Create gene labels
        gene_labels = [self.genes[idx] for idx in gene_indices]

        # Sort samples by response group
        sort_order = {'EXT': 0, 'INT': 1, 'NON': 2}
        sorted_indices = sorted(range(len(sample_groups)), key=lambda i: sort_order.get(sample_groups[i], 3))

        expr_data = expr_data[:, sorted_indices]
        sample_labels = [sample_labels[i] for i in sorted_indices]
        sample_groups = [sample_groups[i] for i in sorted_indices]

        # Z-score normalize the data
        from scipy.stats import zscore
        expr_data_z = np.zeros_like(expr_data)
        for i in range(expr_data.shape[0]):
            row = expr_data[i, :]
            valid_indices = ~np.isnan(row)
            if np.sum(valid_indices) > 1:
                row_valid = row[valid_indices]
                z_scores = zscore(row_valid, nan_policy='omit')
                row_z = np.full_like(row, np.nan)
                row_z[valid_indices] = z_scores
                expr_data_z[i, :] = row_z
            else:
                expr_data_z[i, :] = row - np.nanmean(row) if np.nanmean(row) != 0 else row

        # Create figure with proper dimensions
        fig_width = max(10, len(sample_labels) * 0.4 + 2)
        fig_height = max(10, len(gene_labels) * 0.4 + 2)

        # Create the figure and axes manually to have more control
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create a new clustergrid using clustermap
        cg = sns.clustermap(
            expr_data_z,
            row_cluster=True,
            col_cluster=False,
            cmap='RdBu_r',  # Use RdBu_r as requested
            xticklabels=sample_labels,
            yticklabels=gene_labels,
            figsize=(fig_width, fig_height),
            cbar_kws={'label': 'Expression (Z-score)', 'location': 'right'},  # Move colorbar to right
            cbar_pos=(1, 0.15, 0.03, 0.7),  # Position for the colorbar (right side)
            dendrogram_ratio=0.15,  # Increased for better visibility
            method='average',
            mask=np.isnan(expr_data_z),  # Mask NaN values
            linewidths=0.01,  # Add subtle grid lines
            linecolor='lightgray',
            # Remove col_colors to eliminate the response group bar
        )

        # Customize the appearance
        cg.ax_heatmap.set_facecolor('#f0f0f0')  # Light gray background for missing values

        # Rotate x-axis labels and improve visibility
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize=10, fontweight='bold')

        # Format row dendrogram
        cg.ax_row_dendrogram.spines['top'].set_visible(False)
        cg.ax_row_dendrogram.spines['right'].set_visible(False)
        cg.ax_row_dendrogram.spines['bottom'].set_visible(False)
        cg.ax_row_dendrogram.spines['left'].set_visible(False)

        # Add legend for response groups
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.group_colors['EXT'], label='EXT', edgecolor='black'),
            Patch(facecolor=self.group_colors['INT'], label='INT', edgecolor='black'),
            Patch(facecolor=self.group_colors['NON'], label='NON', edgecolor='black')
        ]
        cg.fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                   fontsize=10, frameon=True, framealpha=0.9, edgecolor='black')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return cg.fig
    
    def analyze_all_timepoints(self, p_threshold=0.05, fc_threshold=0.5):
        """Analyze all timepoints and generate visualizations"""
        print("ANALYZING ALL TIMEPOINTS")
        print("="*70)
        
        timepoints = ['B1W1', 'B2W2', 'B2W4', 'B4W1']
        all_results = {}
        volcano_plots = {}
        heatmap_plots = {}
        
        # Interesting genes to highlight in volcano plots
        highlight_genes = ['CCR3', 'CCL3', 'IL11', 'CSF1', 'HMGB2', 'CD40', 'DEFA1']
        
        for timepoint in timepoints:
            print(f"\n{'-'*30}")
            print(f"ANALYZING {timepoint}")
            print(f"{'-'*30}")
            
            # Analyze timepoint
            tp_results = self.analyze_timepoint(timepoint, p_threshold, fc_threshold)
            
            if tp_results is None or len(tp_results) == 0:
                print(f"⚠️ No valid results for {timepoint}")
                continue
            
            all_results[timepoint] = tp_results
            
            # Create volcano plots
            tp_volcano_plots = {}
            for comp_name, results_df in tp_results.items():
                title = f"{comp_name} at {timepoint}"
                volcano_fig, sig_genes = self.create_volcano_plot(
                    results_df, 
                    title, 
                    p_threshold=p_threshold, 
                    fc_threshold=0.6,  # Set to 0.6 as requested
                    highlight_genes=highlight_genes
                )

                if volcano_fig:
                    filename = f"volcano_{timepoint}_{comp_name}"
                    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
                    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
                    tp_volcano_plots[comp_name] = volcano_fig
            
            # Create heatmap using p-value threshold
            heatmap_fig = self.create_heatmap(timepoint, tp_results, 
                                           p_threshold=p_threshold)
            if heatmap_fig:
                plt.savefig(f"heatmap_{timepoint}.png", dpi=300, bbox_inches='tight')
                plt.savefig(f"heatmap_{timepoint}.pdf", bbox_inches='tight')
                heatmap_plots[timepoint] = heatmap_fig
        
        return {
            'comparison_results': all_results,
            'volcano_plots': volcano_plots,
            'heatmap_plots': heatmap_plots
        }

def run_response_group_comparison(phase1_data, p_threshold=0.05, fc_threshold=0.5):
    """Run response group comparison analysis at individual timepoints"""
    print("STARTING RESPONSE GROUP COMPARISON AT INDIVIDUAL TIMEPOINTS")
    print("="*70)
    print(f"Using p-value threshold: {p_threshold}, fold change threshold: {fc_threshold}")
    
    try:
        # Initialize analyzer
        analyzer = ResponseGroupTimePointAnalyzer(phase1_data)
        
        # Run analysis
        results = analyzer.analyze_all_timepoints(p_threshold, fc_threshold)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        # Summarize results
        print("\nSUMMARY OF SIGNIFICANT GENES (p < 0.05):")
        for timepoint, tp_results in results['comparison_results'].items():
            print(f"\n{timepoint}:")
            for comp_name, results_df in tp_results.items():
                # Count genes by p-value
                p_sig = results_df[results_df['p_value'] < p_threshold]
                p_sig_up = p_sig[p_sig['log2fc'] > 0]
                p_sig_down = p_sig[p_sig['log2fc'] < 0]
                
                print(f"  {comp_name}:")
                print(f"    p < {p_threshold}: {len(p_sig)} genes")
                print(f"      Up: {len(p_sig_up)}, Down: {len(p_sig_down)}")
                
                # List top genes
                if len(p_sig) > 0:
                    top_genes = p_sig.nsmallest(5, 'p_value')['gene'].tolist()
                    print(f"    Top genes: {', '.join(top_genes)}")
        
        return results
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the analysis with standard p-value threshold of 0.05
results = run_response_group_comparison(phase1_data, p_threshold=0.05, fc_threshold=0.6)


# In[15]:


#!/usr/bin/env python
# coding: utf-8

"""
Pooled Analysis of IL-2 Treatment Effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class IL2AnalysisEngine:
    """Complete IL-2 treatment analysis engine"""
    
    def __init__(self, data_file='CICPT_1558_data (2).xlsx'):
        """Initialize with data loading and quality control"""
        self.data_file = data_file
        self.housekeeping_genes = ['PGK1', 'CLTC', 'HPRT1', 'GUSB', 'GAPDH', 'TUBB']
        
        # Load and process data
        self.load_data()
        self.create_sample_mappings()
        self.prepare_expression_matrix()
        self.quality_control_checks()
        
    def load_data(self):
        """Load all data sheets"""
        print("LOADING DATA")
        print("="*50)
        
        try:
            self.sheets = pd.read_excel(self.data_file, sheet_name=None)
            print(f"✓ Loaded sheets: {list(self.sheets.keys())}")
            
            self.sample_info_df = self.sheets['sampleID']
            self.nanostring_df = self.sheets['nanostring']
            
            print(f"✓ Sample info: {self.sample_info_df.shape}")
            print(f"✓ Nanostring data: {self.nanostring_df.shape}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def create_sample_mappings(self):
        """Create comprehensive sample mappings"""
        print("\n CREATING SAMPLE MAPPINGS")
        print("="*50)
        
        # Initialize mappings
        self.sample_to_response = {}
        self.sample_to_patientid = {}
        self.sample_to_timepoint = {}
        
        # Load response group data
        response_df = pd.read_excel(self.data_file, sheet_name="group", header=None)
        
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
        
        print(f"✓ Response groups: {set(mrn_to_response.values())}")
        
        # Map samples to responses and timepoints
        import re
        for _, row in self.sample_info_df.iterrows():
            mrn = row['MRN'] if 'MRN' in row else None
            study_id = row['Study ID'] if 'Study ID' in row else None
            
            samples_str = row.get('Sample #', '')
            samples = str(samples_str).split(',') if pd.notna(samples_str) else []
            
            response = mrn_to_response.get(mrn)
            
            if response:
                for sample in samples:
                    sample = sample.strip()
                    if sample:
                        self.sample_to_response[sample] = response
                        if study_id:
                            self.sample_to_patientid[sample] = study_id
                        
                        # Extract timepoint
                        timepoint_match = re.search(r'\.B(\d+)W(\d+)', sample)
                        if timepoint_match:
                            block = int(timepoint_match.group(1))
                            week = int(timepoint_match.group(2))
                            timepoint = f"B{block}W{week}"
                            
                            self.sample_to_timepoint[sample] = {
                                'timepoint': timepoint,
                                'block': block,
                                'week': week
                            }
        
        # Print mapping summary
        response_counts = {}
        for response in self.sample_to_response.values():
            response_counts[response] = response_counts.get(response, 0) + 1
        
        print(f"✓ Mapped samples: {len(self.sample_to_response)}")
        for response, count in sorted(response_counts.items()):
            print(f"  {response}: {count} samples")
        
        unique_patients = len(set(self.sample_to_patientid.values()))
        print(f"✓ Unique patients: {unique_patients}")
    
    def is_control_gene(self, gene_name):
        """Identify control genes to exclude"""
        if pd.isna(gene_name):
            return True
        gene_str = str(gene_name).upper()
        controls = ['NEG_H', 'NEG_', 'POS_', 'SPIKE', 'CONTROL']
        return any(ctrl in gene_str for ctrl in controls)

    def prepare_expression_matrix(self):
        """Prepare complete expression matrix with ALL genes"""
        print("\n PREPARING EXPRESSION MATRIX")
        print("="*50)

        # Get all genes from nanostring data
        gene_col = self.nanostring_df.columns[0]
        all_genes = self.nanostring_df[gene_col].dropna().tolist()

        print(f"✓ Total genes in dataset: {len(all_genes)}")

        # Remove housekeeping genes AND control genes
        self.genes = []
        for gene in all_genes:
            if gene not in self.housekeeping_genes and not self.is_control_gene(gene):
                self.genes.append(gene)

        housekeeping_count = len([g for g in all_genes if g in self.housekeeping_genes])
        control_count = len([g for g in all_genes if self.is_control_gene(g)])
        total_filtered = len(all_genes) - len(self.genes)

        print(f"✓ Censored {housekeeping_count} housekeeping genes: {self.housekeeping_genes}")
        print(f"✓ Censored {control_count} control genes")
        print(f"✓ Total filtered genes: {total_filtered}")
        print(f"✓ Analyzing {len(self.genes)} genes")

        # Get expression columns that have response mappings
        expr_cols = [col for col in self.nanostring_df.columns 
                    if col in self.sample_to_response]

        print(f"✓ Expression columns with mappings: {len(expr_cols)}")

        # Build expression matrix: genes × samples
        expr_matrix = []
        self.sample_labels = []

        for sample in expr_cols:
            if sample in self.sample_to_response:
                sample_data = []
                for gene in self.genes:
                    gene_rows = self.nanostring_df[self.nanostring_df[gene_col] == gene]
                    if len(gene_rows) > 0:
                        val = pd.to_numeric(gene_rows[sample].iloc[0], errors='coerce')
                        sample_data.append(val if pd.notna(val) else np.nan)
                    else:
                        sample_data.append(np.nan)

                expr_matrix.append(sample_data)
                self.sample_labels.append(sample)

        self.expr_matrix = np.array(expr_matrix).T  # Transpose to genes × samples

        print(f"✓ Expression matrix: {self.expr_matrix.shape} (genes × samples)")

        # Check for missing data
        missing_percent = np.isnan(self.expr_matrix).sum() / self.expr_matrix.size * 100
        print(f"✓ Missing data: {missing_percent:.2f}%")
    
    def quality_control_checks(self):
        """Comprehensive quality control"""
        print("\n QUALITY CONTROL CHECKS")
        print("="*50)
        
        # 1. Check expression value ranges
        finite_values = self.expr_matrix[np.isfinite(self.expr_matrix)]
        print(f"✓ Expression range: {finite_values.min():.2f} to {finite_values.max():.2f}")
        print(f"✓ Expression mean: {finite_values.mean():.2f}")
        print(f"✓ Expression std: {finite_values.std():.2f}")
        
        # Verify log2 transformation
        if 0 <= finite_values.min() and finite_values.max() < 20:
            print("✓ Data appears to be log2 transformed")
        else:
            print("⚠️  Data may not be log2 transformed")
        
        # 2. Check housekeeping gene exclusion
        remaining_hk = [gene for gene in self.housekeeping_genes if gene in self.genes]
        if len(remaining_hk) == 0:
            print("✓ All housekeeping genes successfully excluded")
        else:
            print(f"⚠️  Some housekeeping genes remain: {remaining_hk}")
        
        # 3. Check sample distribution by timepoint
        timepoint_counts = {}
        for sample in self.sample_labels:
            if sample in self.sample_to_timepoint:
                tp = self.sample_to_timepoint[sample]['timepoint']
                timepoint_counts[tp] = timepoint_counts.get(tp, 0) + 1
        
        print(f"✓ Timepoint distribution: {timepoint_counts}")
        
        # 4. Verify IL-2 timepoints are present
        if 'B2W2' in timepoint_counts and 'B2W4' in timepoint_counts:
            print("✓ IL-2 analysis timepoints (B2W2, B2W4) present")
        else:
            print("Missing IL-2 timepoints")
    
    def find_paired_samples(self, timepoint1, timepoint2):
        """Find paired samples between two timepoints"""
        tp1_samples = [s for s in self.sample_labels 
                      if s in self.sample_to_timepoint and 
                      self.sample_to_timepoint[s]['timepoint'] == timepoint1]
        
        tp2_samples = [s for s in self.sample_labels 
                      if s in self.sample_to_timepoint and 
                      self.sample_to_timepoint[s]['timepoint'] == timepoint2]
        
        paired_samples = []
        for s1 in tp1_samples:
            patient1 = self.sample_to_patientid.get(s1)
            for s2 in tp2_samples:
                patient2 = self.sample_to_patientid.get(s2)
                if patient1 == patient2 and patient1 is not None:
                    paired_samples.append((s1, s2, patient1))
                    break
        
        return paired_samples
    
    def analyze_il2_response(self, min_pairs=3, fdr_threshold=0.05, fc_threshold=0.5):
        """Complete IL-2 response analysis (B2W4 vs B2W2)"""
        print("\n IL-2 RESPONSE ANALYSIS (B2W4 vs B2W2)")
        print("="*60)
        
        # Find paired samples
        paired_samples = self.find_paired_samples('B2W2', 'B2W4')
        
        print(f" Found {len(paired_samples)} paired samples:")
        for s1, s2, pid in paired_samples:
            response = self.sample_to_response.get(s1, 'Unknown')
            print(f"  {pid} ({response}): {s1} → {s2}")
        
        if len(paired_samples) < min_pairs:
            print(f" Insufficient paired samples (need ≥{min_pairs})")
            return None
        
        # Analyze each gene
        gene_results = []
        
        print(f"\n Analyzing {len(self.genes)} genes...")
        
        for gene_idx, gene in enumerate(self.genes):
            if gene_idx % 100 == 0:
                print(f"  Processing gene {gene_idx+1}/{len(self.genes)}")
            
            values_b2w2 = []
            values_b2w4 = []
            
            for s1, s2, pid in paired_samples:
                idx1 = self.sample_labels.index(s1)
                idx2 = self.sample_labels.index(s2)
                
                val1 = self.expr_matrix[gene_idx, idx1]  # B2W2
                val2 = self.expr_matrix[gene_idx, idx2]  # B2W4
                
                if pd.notna(val1) and pd.notna(val2):
                    values_b2w2.append(val1)
                    values_b2w4.append(val2)
            
            if len(values_b2w2) >= min_pairs:
                # Paired t-test
                stat, p_val = stats.ttest_rel(values_b2w4, values_b2w2)
                
                # Calculate fold changes
                mean_b2w2 = np.mean(values_b2w2)
                mean_b2w4 = np.mean(values_b2w4)
                log2fc = mean_b2w4 - mean_b2w2  # Correct for log2 data
                fold_change = 2 ** abs(log2fc)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(values_b2w2, ddof=1) + np.var(values_b2w4, ddof=1)) / 2)
                cohens_d = (mean_b2w4 - mean_b2w2) / pooled_std if pooled_std > 0 else 0
                
                gene_results.append({
                    'gene': gene,
                    'log2fc': log2fc,
                    'fold_change': fold_change,
                    'mean_b2w2': mean_b2w2,
                    'mean_b2w4': mean_b2w4,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'n_pairs': len(values_b2w2)
                })
        
        if not gene_results:
            print(" No gene results generated")
            return None
        
        # Convert to DataFrame and apply statistical corrections
        results_df = pd.DataFrame(gene_results)
        
        # FDR correction
        _, fdr_p, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['fdr_p'] = fdr_p
        
        # Bonferroni correction
        _, bonf_p, _, _ = multipletests(results_df['p_value'], method='bonferroni')
        results_df['bonferroni_p'] = bonf_p
        
        # Identify significant genes
        significant_fdr = results_df[
            (results_df['fdr_p'] < fdr_threshold) & 
            (results_df['log2fc'].abs() > fc_threshold)
        ].copy()
        
        significant_bonf = results_df[
            (results_df['bonferroni_p'] < 0.05) & 
            (results_df['log2fc'].abs() > fc_threshold)
        ].copy()
        
        # Sort by significance
        significant_fdr = significant_fdr.sort_values('fdr_p')
        
        print(f"\n STATISTICAL RESULTS:")
        print(f"  Total genes analyzed: {len(results_df):,}")
        print(f"  Significant (FDR < {fdr_threshold}, |log2FC| > {fc_threshold}): {len(significant_fdr)}")
        print(f"  Significant (Bonferroni < 0.05, |log2FC| > {fc_threshold}): {len(significant_bonf)}")
        
        if len(significant_fdr) > 0:
            upregulated = significant_fdr[significant_fdr['log2fc'] > 0]
            downregulated = significant_fdr[significant_fdr['log2fc'] < 0]
            
            print(f"  Upregulated: {len(upregulated)}")
            print(f"  Downregulated: {len(downregulated)}")
            
            print(f"\n TOP UPREGULATED GENES:")
            for _, row in upregulated.head(10).iterrows():
                print(f"  • {row['gene']}: {row['log2fc']:.3f} log2FC, "
                      f"{row['fold_change']:.2f}-fold, FDR={row['fdr_p']:.2e}")
            
            print(f"\n TOP DOWNREGULATED GENES:")
            for _, row in downregulated.head(10).iterrows():
                print(f"  • {row['gene']}: {row['log2fc']:.3f} log2FC, "
                      f"{row['fold_change']:.2f}-fold, FDR={row['fdr_p']:.2e}")
        
        # Store results
        self.il2_results = results_df
        self.il2_significant = significant_fdr
        
        return results_df, significant_fdr
    
    def verify_known_targets(self):
        """Verify known IL-2 target genes"""
        print("\n VERIFYING KNOWN IL-2 TARGETS")
        print("="*50)
        
        # Known IL-2 target genes (should be upregulated)
        il2_targets = {
            'activation': ['IL2RA', 'CD25', 'CD69', 'CD44'],
            'cytotoxic': ['PRF1', 'GZMB', 'GNLY', 'FASLG'],
            'cytokines': ['IFNG', 'TNF', 'IL4', 'IL10'],
            'proliferation': ['MKI67', 'PCNA', 'TOP2A'],
            'transcription': ['STAT5A', 'STAT5B', 'MYC', 'FOXP3']
        }
        
        for category, genes in il2_targets.items():
            print(f"\n{category.upper()} MARKERS:")
            found_genes = 0
            
            for gene in genes:
                if gene in self.genes:
                    gene_data = self.il2_results[self.il2_results['gene'] == gene]
                    if len(gene_data) > 0:
                        row = gene_data.iloc[0]
                        direction = "↑" if row['log2fc'] > 0 else "↓"
                        sig = "***" if row['fdr_p'] < 0.001 else "**" if row['fdr_p'] < 0.01 else "*" if row['fdr_p'] < 0.05 else "ns"
                        
                        print(f"  {gene}: {direction} {row['log2fc']:.3f} log2FC, FDR={row['fdr_p']:.2e} {sig}")
                        found_genes += 1
                    else:
                        print(f"  {gene}: No data")
                else:
                    print(f"  {gene}: Not in dataset")
            
            if found_genes == 0:
                print(f"    No {category} markers found - this is concerning!")
    
    def create_volcano_plot(self, title="IL-2 Treatment Response", save_name="IL2_Response_Volcano_Plot"):
        """Create volcano plot with FDR q-value < 0.05 significance"""
        print("\n CREATING IL-2 VOLCANO PLOT (FDR q < 0.05)")
        print("="*50)
        
        if not hasattr(self, 'il2_results'):
            print(" Run IL-2 analysis first")
            return None
        
        data = self.il2_results.copy()
        
        # Remove any rows with missing data
        data = data.dropna(subset=['log2fc', 'fdr_p'])
        
        # Define thresholds - using FDR q-value
        FC_THRESHOLD = 0.6  # log2(1.5)
        Q_THRESHOLD = 0.05  # FDR q-value
        
        print(f"Thresholds: Log2FC ±{FC_THRESHOLD}, FDR q-value < {Q_THRESHOLD}")
        
        # Create significance categories using FDR
        data['meets_fc_threshold'] = abs(data['log2fc']) >= FC_THRESHOLD
        data['meets_q_threshold'] = data['fdr_p'] < Q_THRESHOLD
        data['meets_both_criteria'] = data['meets_fc_threshold'] & data['meets_q_threshold']
        
        sig_both = data[data['meets_both_criteria']]
        
        print(f"Total genes: {len(data)}")
        print(f"Meeting both criteria (FDR q < {Q_THRESHOLD} + |log2FC| > {FC_THRESHOLD}): {len(sig_both)}")
        
        # CREATE VOLCANO PLOT (using FDR values)
        plt.figure(figsize=(10, 8))
        
        # Plot all points (base layer) - using FDR for y-axis
        plt.scatter(
            data['log2fc'],
            -np.log10(data['fdr_p']),
            alpha=0.6,
            s=40,
            color='#cccccc',
            edgecolor='none',
            zorder=1
        )
        
        # Plot significant points (red) - using FDR
        if len(sig_both) > 0:
            plt.scatter(
                sig_both['log2fc'],
                -np.log10(sig_both['fdr_p']),
                alpha=0.9,
                s=60,
                color='#e41a1c',
                edgecolor='black',
                linewidth=0.5,
                zorder=2
            )
        
        # Add threshold lines - using FDR threshold
        plt.axhline(y=-np.log10(Q_THRESHOLD), color='#e41a1c', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
        plt.axvline(x=FC_THRESHOLD, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
        plt.axvline(x=-FC_THRESHOLD, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
        
        # Add FDR threshold label
        plt.text(
            0.15, 
            -np.log10(Q_THRESHOLD) + 0.05, 
            f'FDR q = {Q_THRESHOLD}', 
            color='#e41a1c',
            fontsize=10,
            fontstyle='italic',
            verticalalignment='bottom'
        )
        
        # Label top significant genes (sorted by FDR)
        if len(sig_both) > 0:
            top_sig = sig_both.nsmallest(min(20, len(sig_both)), 'fdr_p')
            
            for _, row in top_sig.iterrows():
                # Truncate long gene names for better display
                gene_name = row['gene']
                if len(gene_name) > 10:
                    gene_name = gene_name[:10] + "..."
                
                plt.text(
                    row['log2fc'], 
                    -np.log10(row['fdr_p']), 
                    gene_name,
                    fontsize=9,
                    fontweight='bold',
                    ha='center',
                    va='bottom',
                    zorder=3
                )
        
        # Axis labels and styling
        plt.xlabel('Log2 Fold Change (B2W4 vs B2W2)', fontsize=14)
        plt.ylabel('-Log10(FDR q-value)', fontsize=14)
        
        plt.grid(alpha=0.3, linestyle='--', zorder=0)
        
        # Set axis limits using FDR values
        x_range = max(abs(data['log2fc'].min()), abs(data['log2fc'].max()))
        y_max = max(-np.log10(data['fdr_p']))
        
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
    
    def run_complete_analysis(self):
        """Run complete IL-2 analysis pipeline"""
        print(" RUNNING COMPLETE IL-2 ANALYSIS")
        print("="*80)
        
        # Analysis steps
        il2_results, il2_significant = self.analyze_il2_response()
        
        if il2_results is not None:
            self.verify_known_targets()
            significant_volcano_genes = self.create_volcano_plot()
            
            print(f"\n ANALYSIS COMPLETE")
            print("="*50)
            print(f" Results summary:")
            print(f"  • Total genes: {len(self.genes):,}")
            print(f"  • Significant genes (FDR): {len(il2_significant) if il2_significant is not None else 0}")
            print(f"  • Significant genes (volcano): {len(significant_volcano_genes) if significant_volcano_genes is not None else 0}")
            print(f"  • Paired samples: {len(self.find_paired_samples('B2W2', 'B2W4'))}")
            print(f" Files generated:")
            print(f"  • IL2_Response_Volcano_Plot.png/pdf")
            
            return {
                'results': il2_results,
                'significant': il2_significant,
                'significant_volcano': significant_volcano_genes
            }
        else:
            print(" Analysis failed")
            return None

# Initialize and run analysis
print(" STARTING FRESH IL-2 ANALYSIS")
print("="*80)

try:
    # Initialize analyzer
    analyzer = IL2AnalysisEngine()
    
    # Run complete analysis
    analysis_results = analyzer.run_complete_analysis()
    
    print("\n ANALYSIS PIPELINE COMPLETED!")
    
except Exception as e:
    print(f" Error: {e}")
    import traceback
    traceback.print_exc()


# In[13]:


#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive Difference-in-Differences Analysis for All Response Groups and Timepoints
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
import re
warnings.filterwarnings('ignore')

# Set up visualization style
plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

class DifferentialResponseAnalyzer:
    """Analyze differential treatment responses between response groups"""
    
    def __init__(self, phase1_data):
        """Initialize with Phase 1 data"""
        self.sheets = phase1_data['sheets']
        self.sample_to_response = phase1_data['sample_to_response']
        self.sample_to_patientid = phase1_data['sample_to_patientid']
        self.sample_to_timepoint = phase1_data['sample_to_timepoint']
        
        # Response group colors for consistent plotting
        self.group_colors = {
            'EXT': '#d62728',  # Red
            'INT': '#1f77b4',  # Blue  
            'NON': '#7f7f7f'   # Gray
        }
        
        self._prepare_expression_data()
        print("Differential Response Analyzer initialized")
    
    def _prepare_expression_data(self):
        """Prepare expression data from Phase 1, excluding control genes"""
        nanostring_df = self.sheets['nanostring']
        expr_cols = [col for col in nanostring_df.columns if col in self.sample_to_response]
        
        gene_col = nanostring_df.columns[0]
        all_genes = nanostring_df[gene_col].dropna().tolist()
        
        # Filter out control genes (NEG_*, POS_*, etc.)
        control_pattern = re.compile(r'^(NEG|POS)_[A-Z]')
        self.genes = [gene for gene in all_genes if not control_pattern.match(gene)]
        
        print(f"Filtered out {len(all_genes) - len(self.genes)} control genes")
        
        expr_matrix = []
        self.sample_labels = []
        
        for sample in expr_cols:
            if sample in self.sample_to_response:
                sample_data = []
                for gene in self.genes:
                    gene_rows = nanostring_df[nanostring_df[gene_col] == gene]
                    if len(gene_rows) > 0:
                        val = pd.to_numeric(gene_rows[sample].iloc[0], errors='coerce')
                        sample_data.append(val if pd.notna(val) else np.nan)
                    else:
                        sample_data.append(np.nan)
                
                expr_matrix.append(sample_data)
                self.sample_labels.append(sample)
        
        self.expr_matrix = np.array(expr_matrix).T  # genes x samples
        print(f"Expression data prepared: {len(self.genes)} genes, {len(self.sample_labels)} samples")
    
    def analyze_differential_response(self, timepoint1, timepoint2, group1, group2, p_threshold=0.05):
        """
        Analyze how the change from timepoint1 to timepoint2 differs between group1 and group2
        """
        print(f"\nANALYZING DIFFERENTIAL RESPONSE TO {timepoint1} → {timepoint2}")
        print(f"Comparing {group1} vs {group2}")
        print("="*60)
        
        # Find all patients with samples at both timepoints
        group1_patients = {}
        group2_patients = {}
        
        # First, collect all paired samples by patient
        for sample in self.sample_labels:
            if sample not in self.sample_to_response or sample not in self.sample_to_patientid:
                continue
                
            response_group = self.sample_to_response[sample]
            patient_id = self.sample_to_patientid[sample]
            
            if sample not in self.sample_to_timepoint:
                continue
                
            timepoint = self.sample_to_timepoint[sample]['timepoint']
            
            if response_group == group1:
                if patient_id not in group1_patients:
                    group1_patients[patient_id] = {}
                if timepoint in [timepoint1, timepoint2]:
                    group1_patients[patient_id][timepoint] = sample
            
            elif response_group == group2:
                if patient_id not in group2_patients:
                    group2_patients[patient_id] = {}
                if timepoint in [timepoint1, timepoint2]:
                    group2_patients[patient_id][timepoint] = sample
        
        # Filter to patients with both timepoints
        group1_complete = {pid: data for pid, data in group1_patients.items() 
                         if timepoint1 in data and timepoint2 in data}
        
        group2_complete = {pid: data for pid, data in group2_patients.items() 
                         if timepoint1 in data and timepoint2 in data}
        
        print(f"\n{group1} patients with both timepoints: {len(group1_complete)}")
        print(f"{group2} patients with both timepoints: {len(group2_complete)}")
        
        # Check if we have enough patients for analysis
        min_patients = 2  # Set minimum number of patients per group
        if len(group1_complete) < min_patients or len(group2_complete) < min_patients:
            print(f"⚠️ Insufficient paired samples for analysis (minimum {min_patients} per group)")
            return None
        
        # Calculate treatment effect (change from timepoint1 to timepoint2) for each patient
        group1_effects = {}
        group2_effects = {}
        
        # For each gene, calculate the effect for each patient
        gene_results = []
        
        for gene_idx, gene in enumerate(self.genes):
            # Calculate effect for group1
            g1_patient_effects = []
            
            for pid, samples in group1_complete.items():
                s1 = samples[timepoint1]
                s2 = samples[timepoint2]
                
                idx1 = self.sample_labels.index(s1)
                idx2 = self.sample_labels.index(s2)
                
                val1 = self.expr_matrix[gene_idx, idx1]
                val2 = self.expr_matrix[gene_idx, idx2]
                
                if pd.notna(val1) and pd.notna(val2):
                    effect = val2 - val1  # Change from timepoint1 to timepoint2
                    g1_patient_effects.append(effect)
            
            # Calculate effect for group2
            g2_patient_effects = []
            
            for pid, samples in group2_complete.items():
                s1 = samples[timepoint1]
                s2 = samples[timepoint2]
                
                idx1 = self.sample_labels.index(s1)
                idx2 = self.sample_labels.index(s2)
                
                val1 = self.expr_matrix[gene_idx, idx1]
                val2 = self.expr_matrix[gene_idx, idx2]
                
                if pd.notna(val1) and pd.notna(val2):
                    effect = val2 - val1  # Change from timepoint1 to timepoint2
                    g2_patient_effects.append(effect)
            
            # Skip genes with insufficient data
            if len(g1_patient_effects) < min_patients or len(g2_patient_effects) < min_patients:
                continue
            
            # Compare the effects between groups
            try:
                t_stat, p_val = stats.ttest_ind(g1_patient_effects, g2_patient_effects, equal_var=False)
                
                # Calculate difference of differences
                mean_effect1 = np.mean(g1_patient_effects)
                mean_effect2 = np.mean(g2_patient_effects)
                diff_of_diff = mean_effect1 - mean_effect2
                
                gene_results.append({
                    'gene': gene,
                    'p_value': p_val,
                    'diff_of_diff': diff_of_diff,
                    'abs_diff': abs(diff_of_diff),
                    f'{group1}_effect': mean_effect1,
                    f'{group2}_effect': mean_effect2,
                    f'{group1}_n': len(g1_patient_effects),
                    f'{group2}_n': len(g2_patient_effects)
                })
                
            except Exception as e:
                # Skip gene if test fails
                continue
        
        if len(gene_results) == 0:
            print(f"⚠️ No valid results for differential response analysis")
            return None
        
        # Create results dataframe
        results_df = pd.DataFrame(gene_results)
        
        # Apply FDR correction
        _, fdr_p, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['fdr_p'] = fdr_p
        
        # Count significant genes
        sig_genes = results_df[results_df['p_value'] < p_threshold]
        fdr_sig_genes = results_df[results_df['fdr_p'] < p_threshold]
        
        # Direction of effect - positive means group1 has stronger response than group2
        sig_stronger = sig_genes[sig_genes['diff_of_diff'] > 0]
        sig_weaker = sig_genes[sig_genes['diff_of_diff'] < 0]
        
        print(f"\nResults Summary:")
        print(f"  Total genes tested: {len(results_df)}")
        print(f"  Significant genes (p < {p_threshold}): {len(sig_genes)}")
        print(f"  FDR-significant genes (q < {p_threshold}): {len(fdr_sig_genes)}")
        print(f"  Stronger effect in {group1}: {len(sig_stronger)} genes")
        print(f"  Stronger effect in {group2}: {len(sig_weaker)} genes")
        
        # Show top differentially responsive genes
        if len(sig_genes) > 0:
            print(f"\nTop genes with differential response (by p-value):")
            for _, row in sig_genes.nsmallest(10, 'p_value').iterrows():
                direction = f"Stronger in {group1}" if row['diff_of_diff'] > 0 else f"Stronger in {group2}"
                print(f"  {row['gene']}: {direction}, diff={row['diff_of_diff']:.2f}, p={row['p_value']:.4f}")
                print(f"    {group1} effect: {row[f'{group1}_effect']:.2f}, {group2} effect: {row[f'{group2}_effect']:.2f}")
        
        return results_df
    
    def create_diff_of_diff_heatmap(self, results_df, title, timepoints, 
                               max_genes=30, p_threshold=0.05):
        """Create enhanced heatmap for difference-in-differences analysis matching pooled heatmap style"""
        print(f"\nCreating difference-in-differences heatmap for {title}")

        if results_df is None or len(results_df) == 0:
            print(f"⚠️ No data for heatmap: {title}")
            return None

        # Extract group names from column names
        group_cols = [col for col in results_df.columns if col.endswith('_effect')]
        group1 = group_cols[0].replace('_effect', '')
        group2 = group_cols[1].replace('_effect', '')

        # Select significant genes
        sig_genes = results_df[results_df['p_value'] < p_threshold]

        if len(sig_genes) == 0:
            print(f"⚠️ No significant genes for heatmap")
            # Use top genes by p-value instead
            sig_genes = results_df.nsmallest(max_genes, 'p_value')
            print(f"Using top {len(sig_genes)} genes by p-value")
        elif len(sig_genes) > max_genes:
            # Limit to top max_genes
            sig_genes = sig_genes.nsmallest(max_genes, 'p_value')
            print(f"Limiting heatmap to top {max_genes} significant genes")

        # Create heatmap data
        genes = sig_genes['gene'].tolist()
        groups = [group1, group2]

        # Create a dataframe for the heatmap
        heatmap_data = pd.DataFrame(index=genes, columns=groups)

        for gene in genes:
            gene_row = results_df[results_df['gene'] == gene].iloc[0]
            for group in groups:
                heatmap_data.loc[gene, group] = gene_row[f'{group}_effect']

        # Z-score normalize the data
        from scipy.stats import zscore
        heatmap_data_z = heatmap_data.copy()
        for idx in heatmap_data_z.index:
            row = heatmap_data_z.loc[idx, :].values
            valid_indices = ~np.isnan(row)
            if np.sum(valid_indices) > 1:
                row_valid = row[valid_indices]
                z_scores = zscore(row_valid, nan_policy='omit')
                row_z = np.full_like(row, np.nan)
                row_z[valid_indices] = z_scores
                heatmap_data_z.loc[idx, :] = row_z
            else:
                heatmap_data_z.loc[idx, :] = row - np.nanmean(row) if np.nanmean(row) != 0 else row

        # Create figure with proper dimensions
        fig_width = 8  # Smaller width since only 2 columns
        fig_height = max(10, len(genes) * 0.4 + 2)

        # Create the figure
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create a clustergrid using clustermap
        cg = sns.clustermap(
            heatmap_data_z,
            row_cluster=True,
            col_cluster=False,
            cmap='RdBu_r',  # Use RdBu_r as requested
            figsize=(fig_width, fig_height),
            cbar_kws={'label': 'Treatment Effect (Z-score)', 'location': 'right'},  # Move colorbar to right
            cbar_pos=(0.98, 0.15, 0.03, 0.7),  # Position for the colorbar (right side)
            dendrogram_ratio=0.15,  # Increased for better visibility
            method='average',
            mask=np.isnan(heatmap_data_z.values),  # Mask NaN values
            linewidths=0.01,  # Add subtle grid lines
            linecolor='lightgray',
            xticklabels=True,
            yticklabels=True
        )

        # Color the column labels based on response group
        for i, col in enumerate(heatmap_data_z.columns):
            color = self.group_colors.get(col, 'black')
            cg.ax_heatmap.get_xticklabels()[i].set_color(color)
            cg.ax_heatmap.get_xticklabels()[i].set_fontweight('bold')
            cg.ax_heatmap.get_xticklabels()[i].set_fontsize(12)

        # Customize the appearance
        cg.ax_heatmap.set_facecolor('#f0f0f0')  # Light gray background for missing values

        # Make gene labels larger and bold
        plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize=10, fontweight='bold')

        # Format row dendrogram
        cg.ax_row_dendrogram.spines['top'].set_visible(False)
        cg.ax_row_dendrogram.spines['right'].set_visible(False)
        cg.ax_row_dendrogram.spines['bottom'].set_visible(False)
        cg.ax_row_dendrogram.spines['left'].set_visible(False)

        # Add title with better positioning and formatting
        full_title = f"Differential Treatment Effect: {title}\n({timepoints[0]} → {timepoints[1]}, p < {p_threshold})"
        cg.fig.suptitle(full_title, fontsize=16, y=0.98, fontweight='bold')

        # Add legend for response groups
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.group_colors.get(group1, 'gray'), label=group1, edgecolor='black'),
            Patch(facecolor=self.group_colors.get(group2, 'gray'), label=group2, edgecolor='black'),
        ]
        cg.fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                   fontsize=10, frameon=True, framealpha=0.9, edgecolor='black')

        # Add annotation for interpretation
        annotation_text = f"Values represent treatment effect (log2FC) from {timepoints[0]} to {timepoints[1]}"
        cg.fig.text(0.02, 0.01, annotation_text, fontsize=8, ha='left', va='bottom')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return cg.fig
    
    def run_all_analyses(self):
        """Run differential response analyses for all possible combinations"""
        print("RUNNING COMPREHENSIVE DIFFERENTIAL RESPONSE ANALYSES")
        print("="*80)
        
        # Define all possible transitions
        timepoint_pairs = [
            ('B1W1', 'B2W2', 'Pembrolizumab Effect'),   # Pembrolizumab effect
            ('B2W2', 'B2W4', 'IL-2 Effect'),            # IL-2 effect
            ('B2W4', 'B4W1', 'Post-IL-2 Effect'),       # Post-IL-2 effect
            ('B1W1', 'B4W1', 'Overall Effect')          # Overall treatment effect
        ]
        
        # Define all possible group comparisons
        group_pairs = [
            ('EXT', 'NON', 'EXT vs NON'),
            ('INT', 'NON', 'INT vs NON'),
            ('EXT', 'INT', 'EXT vs INT')
        ]
        
        all_results = {}
        
        # Test all combinations
        for tp1, tp2, tp_label in timepoint_pairs:
            for g1, g2, g_label in group_pairs:
                title = f"{tp_label}: {g_label}"
                
                print(f"\n{'-'*80}")
                print(f"ANALYZING {title}")
                print(f"{'-'*80}")
                
                results = self.analyze_differential_response(tp1, tp2, g1, g2)
                
                if results is not None and len(results) > 0:
                    # Create a key for storing results
                    key = f"{tp_label.replace(' ', '_')}_{g_label.replace(' ', '_')}"
                    all_results[key] = results
        
        return all_results

# Run the analysis
def run_comprehensive_differential_response_analysis(phase1_data):
    """Run differential response analysis for all combinations"""
    print("STARTING COMPREHENSIVE DIFFERENTIAL RESPONSE ANALYSIS")
    print("="*80)
    
    try:
        # Initialize analyzer
        analyzer = DifferentialResponseAnalyzer(phase1_data)
        
        # Run analyses
        results = analyzer.run_all_analyses()
        
        print("\n" + "="*80)
        print("DIFFERENTIAL RESPONSE ANALYSIS COMPLETE")
        print("="*80)
        
        # Summary of findings
        print("\nSUMMARY OF SIGNIFICANT FINDINGS (p < 0.05):")
        
        for title, df in results.items():
            sig_genes = df[df['p_value'] < 0.05]
            if len(sig_genes) > 0:
                print(f"\n{title.replace('_', ' ')}:")
                print(f"  Significant genes (p < 0.05): {len(sig_genes)}")
                
                # Direction of effects
                group_cols = [col for col in df.columns if col.endswith('_effect')]
                group1 = group_cols[0].replace('_effect', '')
                group2 = group_cols[1].replace('_effect', '')
                
                stronger_g1 = sig_genes[sig_genes['diff_of_diff'] > 0]
                stronger_g2 = sig_genes[sig_genes['diff_of_diff'] < 0]
                
                print(f"  Stronger in {group1}: {len(stronger_g1)}")
                print(f"  Stronger in {group2}: {len(stronger_g2)}")
                
                # List top genes
                top_genes = sig_genes.nsmallest(5, 'p_value')['gene'].tolist()
                print(f"  Top genes: {', '.join(top_genes)}")
                
                # Check for FDR-significant genes
                fdr_sig = df[df['fdr_p'] < 0.05]
                if len(fdr_sig) > 0:
                    fdr_genes = fdr_sig['gene'].tolist()
                    print(f"  FDR-significant genes: {len(fdr_sig)}")
                    print(f"  FDR-significant genes: {', '.join(fdr_genes[:5])}")
        
        return results
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the comprehensive analysis
diff_results = run_comprehensive_differential_response_analysis(phase1_data)


# In[ ]:




