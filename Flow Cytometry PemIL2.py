#!/usr/bin/env python
# coding: utf-8

# In[13]:


#!/usr/bin/env python
# coding: utf-8

"""
Flow Cytometry Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, levene, shapiro
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting
plt.style.use('default')
sns.set_context("paper", font_scale=1.1)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

class FlowAnalyzer:
    
    def __init__(self, flow_df_with_metadata, verbose=True):
        """
        Initialize with pre-loaded and mapped flow data
        
        Parameters:
        - flow_df_with_metadata: DataFrame with flow data and response_group mapping
        """
        self.verbose = verbose
        self.flow_df = flow_df_with_metadata.copy()
        self.qc_metrics = {}
        self.analysis_results = {}
        
        # Define measurement types
        self.measurement_types = {
            'frequency_parent': {
                'column': '% of parent',
                'name': 'Cell Frequency (% Parent)',
                'interpretation': 'Proportion of cells within parent gate',
                'typical_range': (0, 100),
                'log_transform': False,
                'test_type': 'frequency'
            },
            'frequency_all': {
                'column': '% of all Cells', 
                'name': 'Cell Frequency (% Total)',
                'interpretation': 'Proportion of cells in total sample',
                'typical_range': (0, 100),
                'log_transform': False,
                'test_type': 'frequency'
            },
            'mfi_median': {
                'column': 'Median',
                'name': 'Median Fluorescence Intensity',
                'interpretation': 'Expression level per cell (median)',
                'typical_range': (0, 100000),
                'log_transform': True,
                'test_type': 'intensity'
            },
            'mfi_geometric': {
                'column': 'Geometric Mean',
                'name': 'Geometric Mean Fluorescence Intensity', 
                'interpretation': 'Expression level per cell (geometric mean)',
                'typical_range': (0, 100000),
                'log_transform': True,
                'test_type': 'intensity'
            }
        }
        
        # Define immune markers
        self.immune_markers = {
            'CD4': 'CD4-BV480',
            'CD8': 'CD8-A700', 
            'CD19': 'CD19-BUV496',
            'CD16': 'CD16-A647',
            'CD25': 'CD25-BB515',
            'CD15': 'CD15-BV711',
            'CD33': 'CD33-PE',
            'CD14': 'CD14-BUV805',
            'CD11b': 'CD11b-BB700'
        }
        
        if self.verbose:
            print("FLOW CYTOMETRY ANALYSIS")
            print("="*60)
            print(f"Input data: {self.flow_df.shape}")
            print(f"Available columns: {list(self.flow_df.columns)}")
        
        self.perform_quality_control()
    
    def perform_quality_control(self):
        """Comprehensive quality control assessment"""
        if self.verbose:
            print("\nQUALITY CONTROL ASSESSMENT")
            print("-"*30)
        
        # Check data completeness
        self.qc_metrics['total_samples'] = len(self.flow_df)
        self.qc_metrics['samples_with_response'] = len(self.flow_df[self.flow_df['response_group'].notna()])
        
        # Response group distribution
        if 'response_group' in self.flow_df.columns:
            response_dist = self.flow_df['response_group'].value_counts()
            self.qc_metrics['response_distribution'] = response_dist.to_dict()
            
            if self.verbose:
                print(f"Response group distribution:")
                for group, count in response_dist.items():
                    print(f"  {group}: n={count}")
        
        # Parameter availability
        available_params = self.flow_df['Parameter'].value_counts()
        self.qc_metrics['parameter_counts'] = available_params.to_dict()
        
        if self.verbose:
            print(f"\nParameter availability:")
            for param, count in available_params.head(10).items():
                print(f"  {param}: n={count}")
        
        # Check measurement column availability
        measurement_availability = {}
        for mtype, info in self.measurement_types.items():
            col = info['column']
            if col in self.flow_df.columns:
                non_null = self.flow_df[col].notna().sum()
                measurement_availability[mtype] = {
                    'available': True,
                    'non_null_count': non_null,
                    'completeness': non_null / len(self.flow_df)
                }
            else:
                measurement_availability[mtype] = {'available': False}
        
        self.qc_metrics['measurement_availability'] = measurement_availability
        
        if self.verbose:
            print(f"\nMeasurement type availability:")
            for mtype, info in measurement_availability.items():
                if info['available']:
                    print(f"  ✓ {mtype}: {info['completeness']:.1%} complete")
                else:
                    print(f"  ❌ {mtype}: Not available")
    
    def check_statistical_assumptions(self, data_groups, measurement_type):
        """Check statistical assumptions for appropriate test selection"""
        
        assumptions = {
            'normality': {},
            'equal_variance': {},
            'sample_sizes': {},
            'recommended_test': None
        }
        
        # Check sample sizes
        for group, values in data_groups.items():
            assumptions['sample_sizes'][group] = len(values)
        
        min_n = min(assumptions['sample_sizes'].values())
        
        # Normality tests (if n >= 8 for meaningful results)
        if min_n >= 8:
            for group, values in data_groups.items():
                if len(values) >= 8:
                    _, p_shapiro = shapiro(values[:50])  # Limit to 50 for computational efficiency
                    assumptions['normality'][group] = {
                        'shapiro_p': p_shapiro,
                        'normal': p_shapiro > 0.05
                    }
        
        # Equal variance test (Levene's test)
        if len(data_groups) >= 2 and min_n >= 5:
            group_values = list(data_groups.values())
            if len(group_values) >= 2:
                try:
                    _, p_levene = levene(*group_values)
                    assumptions['equal_variance'] = {
                        'levene_p': p_levene,
                        'equal_var': p_levene > 0.05
                    }
                except:
                    assumptions['equal_variance'] = {'equal_var': None}
        
        # Recommend test based on assumptions and data type
        if measurement_type in ['frequency_parent', 'frequency_all']:
            # For proportions/percentages, always use non-parametric
            assumptions['recommended_test'] = 'mann_whitney'
        else:
            # For continuous data, choose based on assumptions
            if min_n < 15:
                assumptions['recommended_test'] = 'mann_whitney'
            else:
                # Check if normality and equal variance assumptions are met
                all_normal = all(info.get('normal', False) for info in assumptions['normality'].values())
                equal_var = assumptions['equal_variance'].get('equal_var', False)
                
                if all_normal and equal_var:
                    assumptions['recommended_test'] = 'welch_t'  # More robust than student's t
                else:
                    assumptions['recommended_test'] = 'mann_whitney'
        
        return assumptions
    
    def perform_statistical_test(self, group1_data, group2_data, test_type='mann_whitney'):
        """Perform appropriate statistical test"""
        
        try:
            if test_type == 'mann_whitney':
                statistic, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                test_name = 'Mann-Whitney U'
                
            elif test_type == 'welch_t':
                statistic, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                test_name = "Welch's t-test"
                
            elif test_type == 'student_t':
                statistic, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=True)
                test_name = "Student's t-test"
                
            else:
                # Default to Mann-Whitney
                statistic, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                test_name = 'Mann-Whitney U (default)'
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'test_used': test_name
            }
            
        except Exception as e:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'test_used': f'Error: {str(e)}'
            }
    
    def calculate_effect_size(self, group1_data, group2_data, measurement_type):
        """Calculate appropriate effect size measures"""
        
        n1, n2 = len(group1_data), len(group2_data)
        mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
        
        effect_sizes = {}
        
        # Fold change (always calculated)
        if mean2 > 0:
            effect_sizes['fold_change'] = mean1 / mean2
            effect_sizes['log2_fold_change'] = np.log2(effect_sizes['fold_change'])
        else:
            effect_sizes['fold_change'] = np.nan
            effect_sizes['log2_fold_change'] = np.nan
        
        # Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1_data, ddof=1) + 
                             (n2 - 1) * np.var(group2_data, ddof=1)) / (n1 + n2 - 2))
        if pooled_std > 0:
            effect_sizes['cohens_d'] = (mean1 - mean2) / pooled_std
        else:
            effect_sizes['cohens_d'] = np.nan
        
        # Glass's delta (using group2 as control)
        std2 = np.std(group2_data, ddof=1)
        if std2 > 0:
            effect_sizes['glass_delta'] = (mean1 - mean2) / std2
        else:
            effect_sizes['glass_delta'] = np.nan
        
        # Cliff's delta (rank-based effect size)
        try:
            from scipy.stats import ranksums
            # Approximate Cliff's delta using rank sum
            n_total = n1 + n2
            all_data = np.concatenate([group1_data, group2_data])
            ranks = stats.rankdata(all_data)
            rank_sum1 = np.sum(ranks[:n1])
            U1 = rank_sum1 - n1 * (n1 + 1) / 2
            cliff_delta = (2 * U1) / (n1 * n2) - 1
            effect_sizes['cliff_delta'] = cliff_delta
        except:
            effect_sizes['cliff_delta'] = np.nan
        
        return effect_sizes
    
    def analyze_single_marker(self, marker_name, measurement_type, min_samples=5):
        """Comprehensive analysis of a single marker with one measurement type"""
        
        if marker_name not in self.immune_markers:
            return None
        
        if measurement_type not in self.measurement_types:
            return None
        
        parameter_name = self.immune_markers[marker_name]
        measurement_info = self.measurement_types[measurement_type]
        measurement_col = measurement_info['column']
        
        # Filter data for this marker
        marker_data = self.flow_df[self.flow_df['Parameter'] == parameter_name].copy()
        
        if len(marker_data) == 0:
            return None
        
        # Prepare data by response group
        group_data = {}
        for group in ['EXT', 'INT', 'NON']:
            group_subset = marker_data[marker_data['response_group'] == group]
            values = pd.to_numeric(group_subset[measurement_col], errors='coerce').dropna()
            
            # Apply log transformation if specified
            if measurement_info['log_transform'] and len(values) > 0:
                # Add small constant to handle zeros
                values = np.log2(values + 1)
            
            group_data[group] = values
        
        # Check minimum sample sizes
        sample_sizes = {group: len(values) for group, values in group_data.items()}
        if any(n < min_samples for n in sample_sizes.values()):
            if self.verbose:
                print(f"❌ {marker_name} ({measurement_type}): Insufficient samples {sample_sizes}")
            return None
        
        if self.verbose:
            print(f"\n{marker_name} ({measurement_type}): {sample_sizes}")
        
        # Check statistical assumptions
        assumptions = self.check_statistical_assumptions(group_data, measurement_type)
        
        # Perform pairwise comparisons
        comparisons = [('EXT', 'NON'), ('INT', 'NON'), ('EXT', 'INT')]
        comparison_results = {}
        all_p_values = []
        
        for group1, group2 in comparisons:
            if len(group_data[group1]) >= min_samples and len(group_data[group2]) >= min_samples:
                
                # Perform statistical test
                test_result = self.perform_statistical_test(
                    group_data[group1], 
                    group_data[group2], 
                    assumptions['recommended_test']
                )
                
                # Calculate effect sizes
                effect_sizes = self.calculate_effect_size(
                    group_data[group1], 
                    group_data[group2], 
                    measurement_type
                )
                
                # Descriptive statistics
                desc_stats = {
                    'mean_group1': np.mean(group_data[group1]),
                    'mean_group2': np.mean(group_data[group2]),
                    'median_group1': np.median(group_data[group1]),
                    'median_group2': np.median(group_data[group2]),
                    'std_group1': np.std(group_data[group1], ddof=1),
                    'std_group2': np.std(group_data[group2], ddof=1),
                    'n_group1': len(group_data[group1]),
                    'n_group2': len(group_data[group2])
                }
                
                comparison_results[f'{group1}_vs_{group2}'] = {
                    **test_result,
                    **effect_sizes,
                    **desc_stats,
                    'assumptions': assumptions
                }
                
                all_p_values.append(test_result['p_value'])
        
        # Apply multiple testing correction
        if len(all_p_values) > 0:
            try:
                rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
                    all_p_values, alpha=0.05, method='fdr_bh'
                )
                
                # Add corrected p-values to results
                for i, (comparison_name, _) in enumerate(comparison_results.items()):
                    comparison_results[comparison_name]['p_adjusted'] = p_adjusted[i]
                    comparison_results[comparison_name]['significant_raw'] = all_p_values[i] < 0.05
                    comparison_results[comparison_name]['significant_fdr'] = rejected[i]
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: FDR correction failed: {e}")
                
                # Fallback: use raw p-values
                for comparison_name in comparison_results.keys():
                    comparison_results[comparison_name]['p_adjusted'] = comparison_results[comparison_name]['p_value']
                    comparison_results[comparison_name]['significant_fdr'] = comparison_results[comparison_name]['p_value'] < 0.05
        
        return {
            'marker': marker_name,
            'measurement_type': measurement_type,
            'parameter': parameter_name,
            'sample_sizes': sample_sizes,
            'assumptions': assumptions,
            'comparisons': comparison_results
        }
    
    def run_comprehensive_analysis(self, min_samples=5):
        """Run comprehensive analysis across all markers and measurement types"""
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("COMPREHENSIVE STATISTICAL ANALYSIS")
            print("="*60)
        
        self.analysis_results = {}
        
        # Test each measurement type
        for measurement_type, measurement_info in self.measurement_types.items():
            
            if not self.qc_metrics['measurement_availability'][measurement_type]['available']:
                if self.verbose:
                    print(f"\n❌ Skipping {measurement_type}: Not available")
                continue
            
            if self.verbose:
                print(f"\n{measurement_type.upper().replace('_', ' ')} ANALYSIS")
                print(f"Measurement: {measurement_info['name']}")
                print(f"Interpretation: {measurement_info['interpretation']}")
                print("-" * 50)
            
            measurement_results = {}
            
            # Analyze each immune marker
            for marker_name in self.immune_markers.keys():
                
                result = self.analyze_single_marker(marker_name, measurement_type, min_samples)
                
                if result is not None:
                    measurement_results[marker_name] = result
                    
                    # Print key results
                    if self.verbose:
                        print(f"\n{marker_name} Results:")
                        for comp_name, comp_data in result['comparisons'].items():
                            fc = comp_data['fold_change']
                            p_val = comp_data['p_value']
                            p_adj = comp_data.get('p_adjusted', np.nan)
                            test_used = comp_data['test_used']
                            
                            # Significance markers
                            sig_raw = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                            sig_fdr = " [FDR]" if comp_data.get('significant_fdr', False) else ""
                            
                            print(f"  {comp_name}: FC={fc:.3f}, p={p_val:.2e} {sig_raw}, "
                                  f"FDR q={p_adj:.2e}{sig_fdr} ({test_used})")
            
            self.analysis_results[measurement_type] = measurement_results
        
        return self.analysis_results
    
    def create_results_summary(self):
        """Create comprehensive results summary"""
        
        if not self.analysis_results:
            print("No analysis results available. Run comprehensive analysis first.")
            return None
        
        summary_data = []
        
        for measurement_type, measurement_results in self.analysis_results.items():
            for marker_name, marker_results in measurement_results.items():
                for comparison_name, comparison_data in marker_results['comparisons'].items():
                    
                    summary_data.append({
                        'Measurement_Type': measurement_type,
                        'Marker': marker_name,
                        'Comparison': comparison_name,
                        'N_Group1': comparison_data['n_group1'],
                        'N_Group2': comparison_data['n_group2'],
                        'Mean_Group1': comparison_data['mean_group1'],
                        'Mean_Group2': comparison_data['mean_group2'],
                        'Fold_Change': comparison_data['fold_change'],
                        'Log2_Fold_Change': comparison_data['log2_fold_change'],
                        'Cohens_D': comparison_data['cohens_d'],
                        'Cliff_Delta': comparison_data['cliff_delta'],
                        'Test_Used': comparison_data['test_used'],
                        'P_Value': comparison_data['p_value'],
                        'P_Adjusted': comparison_data.get('p_adjusted', np.nan),
                        'Significant_Raw': comparison_data.get('significant_raw', False),
                        'Significant_FDR': comparison_data.get('significant_fdr', False)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by significance and effect size
        summary_df['abs_log2fc'] = abs(summary_df['Log2_Fold_Change'])
        summary_df = summary_df.sort_values(['Significant_FDR', 'P_Adjusted', 'abs_log2fc'], 
                                          ascending=[False, True, False])
        
        return summary_df
    
    def print_top_findings(self, n_top=10):
        """Print top findings with biological interpretation"""
        
        summary_df = self.create_results_summary()
        
        if summary_df is None:
            return
        
        print(f"\n{'='*80}")
        print(f"TOP {n_top} FINDINGS (FDR-corrected)")
        print("="*80)
        
        # Focus on FDR-significant results first
        fdr_significant = summary_df[summary_df['Significant_FDR'] == True]
        
        if len(fdr_significant) > 0:
            print(f"\nFDR-SIGNIFICANT RESULTS (q<0.05):")
            print(f"{'Marker':<8} {'Measurement':<15} {'Comparison':<12} {'FC':<8} {'FDR q':<12} {'Effect':<8}")
            print("-" * 80)
            
            for _, row in fdr_significant.head(n_top).iterrows():
                marker = row['Marker']
                mtype = row['Measurement_Type'].replace('_', ' ').title()
                comparison = row['Comparison'].replace('_vs_', ' vs ')
                fc = row['Fold_Change']
                fdr_q = row['P_Adjusted']
                effect = "Large" if abs(row['Cohens_D']) > 0.8 else "Medium" if abs(row['Cohens_D']) > 0.5 else "Small"
                
                print(f"{marker:<8} {mtype:<15} {comparison:<12} {fc:<8.3f} {fdr_q:<12.2e} {effect:<8}")
        else:
            print("No FDR-significant results found")
            
            # Show top raw significant results
            raw_significant = summary_df[summary_df['Significant_Raw'] == True]
            if len(raw_significant) > 0:
                print(f"\nTOP RAW SIGNIFICANT RESULTS (p<0.05, not FDR-corrected):")
                print(f"{'Marker':<8} {'Measurement':<15} {'Comparison':<12} {'FC':<8} {'p-value':<12}")
                print("-" * 70)
                
                for _, row in raw_significant.head(n_top).iterrows():
                    marker = row['Marker']
                    mtype = row['Measurement_Type'].replace('_', ' ').title()
                    comparison = row['Comparison'].replace('_vs_', ' vs ')
                    fc = row['Fold_Change']
                    p_val = row['P_Value']
                    
                    print(f"{marker:<8} {mtype:<15} {comparison:<12} {fc:<8.3f} {p_val:<12.2e}")

# Usage example and main execution
def run_flow_analysis(flow_df_with_metadata):
    """
    Main function to run flow cytometry analysis
    
    Parameters:
    - flow_df_with_metadata: DataFrame from previous analysis with response_group mapping
    """
    
    # Initialize analyzer
    analyzer = FlowAnalyzer(flow_df_with_metadata, verbose=True)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(min_samples=5)
    
    # Create and save summary
    summary_df = analyzer.create_results_summary()
    if summary_df is not None:
        summary_df.to_csv('flow_analysis_results.csv', index=False)
        print(f"\n✓ Results saved to 'flow_analysis_results.csv'")
    
    # Print top findings
    analyzer.print_top_findings(n_top=15)
    
    return analyzer, results, summary_df

if __name__ == "__main__":
    print("Flow Cytometry Analysis Module")
    print("Usage: analyzer, results, summary = run_flow_analysis(flow_df_with_timepoints)")


# In[17]:


#!/usr/bin/env python
# coding: utf-8

"""
Flow Data Preparation
"""

import pandas as pd
import numpy as np
import re

def prepare_flow_data_for_analysis(data_file='CICPT_1558_data (3).xlsx'):
    """
    Load and prepare flow data with response group mapping
    
    Parameters:
    - data_file: Path to Excel file with flow data
    
    Returns:
    - flow_df_ready: DataFrame ready for analysis
    """
    
    print("PREPARING FLOW DATA FOR ANALYSIS")
    print("="*50)
    
    try:
        # Load all necessary sheets
        print("Loading data sheets...")
        sample_info_df = pd.read_excel(data_file, sheet_name='sampleID', header=None)
        flow_df = pd.read_excel(data_file, sheet_name=2)  # Flow data sheet
        response_df = pd.read_excel(data_file, sheet_name='group', header=None)
        
        print(f"✓ Flow data: {flow_df.shape}")
        print(f"✓ Sample info: {sample_info_df.shape}")
        print(f"✓ Response groups: {response_df.shape}")
        
    except Exception as e:
        print(f" Error loading data: {e}")
        return None
    
    # Create MRN to response mapping
    print("\nCreating response group mapping...")
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
    
    print(f"✓ MRN mappings: {len(mrn_to_response)}")
    print(f"✓ Response groups: {set(mrn_to_response.values())}")
    
    # Create Study ID to response mapping
    print("Creating Study ID mapping...")
    studyid_to_response = {}
    
    for i in range(len(sample_info_df)):
        mrn = sample_info_df.iloc[i, 0]  # Column A
        study_id = sample_info_df.iloc[i, 1]  # Column B
        
        if pd.isna(mrn) or pd.isna(study_id):
            continue
        
        # Try different MRN formats
        mrn_variants = [mrn]
        if str(mrn).replace('.','').replace('-','').isdigit():
            mrn_variants.extend([int(float(mrn)), str(int(float(mrn)))])
        mrn_variants.append(str(mrn))
        mrn_variants = [m for m in set(mrn_variants) if m is not None]
        
        response = None
        for mrn_variant in mrn_variants:
            response = mrn_to_response.get(mrn_variant)
            if response:
                break
        
        if response and study_id:
            studyid_to_response[study_id] = response
    
    print(f"✓ Study ID mappings: {len(studyid_to_response)}")
    
    # Add response groups to flow data
    print("Adding response groups to flow data...")
    flow_df['response_group'] = None
    flow_df['patient_id'] = None
    
    mapped_samples = 0
    for idx, row in flow_df.iterrows():
        filename = str(row.get('Filename', ''))
        
        # Extract patient number from filename
        # Pattern: Sample_PT 01-1_037.fcs -> patient 01
        pt_match = re.search(r'PT\s*(\d+)', filename)
        
        if pt_match:
            patient_num = pt_match.group(1).zfill(3)  # Pad to 3 digits
            study_id = f"01-{patient_num}"
            
            response = studyid_to_response.get(study_id)
            if response:
                flow_df.at[idx, 'response_group'] = response
                flow_df.at[idx, 'patient_id'] = study_id
                mapped_samples += 1
    
    print(f"✓ Mapped samples: {mapped_samples}")
    
    # Show response group distribution
    response_counts = flow_df['response_group'].value_counts()
    print("Response group distribution:")
    for response, count in response_counts.items():
        print(f"  {response}: {count}")
    
    print(f"\n✓ Flow data ready for analysis: {flow_df.shape}")
    return flow_df

# Execute the data preparation
print("FLOW DATA PREPARATION")
print("="*60)

# Prepare the data
flow_df_ready = prepare_flow_data_for_analysis('CICPT_1558_data (3).xlsx')

if flow_df_ready is not None:
    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print("✓ Flow data loaded and mapped")
    print("✓ Response groups assigned")
    print("✓ Ready for analysis")
    
    # Show data summary
    print(f"\nData Summary:")
    print(f"  Total samples: {len(flow_df_ready)}")
    print(f"  Samples with response groups: {flow_df_ready['response_group'].notna().sum()}")
    print(f"  Available columns: {list(flow_df_ready.columns)}")
    
    # Check available measurement columns
    measurement_cols = ['% of parent', '% of all Cells', 'Median', 'Geometric Mean']
    print(f"\nAvailable measurements:")
    for col in measurement_cols:
        if col in flow_df_ready.columns:
            non_null = flow_df_ready[col].notna().sum()
            print(f"  ✓ {col}: {non_null} values")
        else:
            print(f"   {col}: Not available")
    
    # Check parameter distribution
    param_counts = flow_df_ready['Parameter'].value_counts()
    print(f"\nTop parameters:")
    for param, count in param_counts.head(10).items():
        print(f"  {param}: {count}")
    
    print(f"\n{'='*60}")
    print("READY TO RUN ANALYSIS")
    print("="*60)
    print("You can now run:")
    print("analyzer, results, summary_df = run_flow_analysis(flow_df_ready)")
    
else:
    print(" Data preparation failed")
    print("Please check that 'CICPT_1558_data (3).xlsx' is in your working directory")
    print("and contains the expected sheets: 'sampleID', 'group', and sheet index 2 for flow data")


# In[18]:


analyzer, results, summary_df = run_flow_analysis(flow_df_ready)


# In[28]:


def create_main_cd16_figure(flow_df, measurement_col='% of all Cells'):
    """
    Create main publication figure for CD16+ NK cells using % of All Cells
    Improved layout with better spacing and significance filtering
    """
    
    print(f"Creating main CD16+ NK cell figure using {measurement_col}...")
    
    # Get CD16 data
    cd16_data = flow_df[flow_df['Parameter'] == 'CD16-A647']
    
    if len(cd16_data) == 0:
        print("❌ No CD16 data found")
        return
    
    # Check if measurement column exists
    if measurement_col not in cd16_data.columns:
        print(f"❌ Column '{measurement_col}' not found")
        print(f"Available columns: {list(cd16_data.columns)}")
        return
    
    # Prepare data for plotting
    plot_data = []
    for _, row in cd16_data.iterrows():
        value = pd.to_numeric(row[measurement_col], errors='coerce')
        if pd.notna(value) and pd.notna(row['response_group']):
            plot_data.append({
                'value': value,
                'response_group': row['response_group']
            })
    
    if len(plot_data) == 0:
        print("❌ No valid plot data")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    print(f"Data summary:")
    for group in ['EXT', 'INT', 'NON']:
        n = len(plot_df[plot_df['response_group'] == group])
        mean_val = plot_df[plot_df['response_group'] == group]['value'].mean()
        print(f"  {group}: n={n}, mean={mean_val:.3f}")
    
    # Perform statistical tests
    def perform_statistical_tests(data):
        """Perform pairwise statistical tests with FDR correction"""
        
        comparisons = [('INT', 'NON'), ('EXT', 'NON'), ('EXT', 'INT')]
        results = {}
        p_values = []
        comp_names = []
        
        for group1, group2 in comparisons:
            data1 = data[data['response_group'] == group1]['value'].dropna()
            data2 = data[data['response_group'] == group2]['value'].dropna()
            
            if len(data1) >= 3 and len(data2) >= 3:
                try:
                    statistic, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    # Calculate fold change
                    mean1 = np.mean(data1)
                    mean2 = np.mean(data2)
                    fold_change = mean1 / mean2 if mean2 > 0 else np.nan
                    
                    results[f'{group1}_vs_{group2}'] = {
                        'p_value': p_val,
                        'fold_change': fold_change,
                        'n1': len(data1),
                        'n2': len(data2)
                    }
                    p_values.append(p_val)
                    comp_names.append(f'{group1}_vs_{group2}')
                    
                except Exception as e:
                    print(f"Error in statistical test for {group1} vs {group2}: {e}")
        
        # FDR correction
        if len(p_values) > 0:
            try:
                _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
                for i, comp_name in enumerate(comp_names):
                    results[comp_name]['p_adjusted'] = p_adj[i]
                    print(f"{comp_name}: FC={results[comp_name]['fold_change']:.2f}, p={p_values[i]:.2e}, FDR q={p_adj[i]:.2e}")
            except:
                for comp_name in comp_names:
                    if comp_name in results:
                        results[comp_name]['p_adjusted'] = results[comp_name]['p_value']
        
        return results
    
    # Run statistical tests
    stats_results = perform_statistical_tests(plot_df)
    
    def format_p_and_fdr(p_val, fdr_val):
        """Format p-value and FDR for display"""
        if pd.isna(p_val):
            return "ns", 'gray', 'normal'
            
        if p_val < 0.001:
            p_text = "p<0.001"
        else:
            p_text = f"p={p_val:.3f}"
        
        # Show FDR if significant (< 0.05)
        if pd.notna(fdr_val) and fdr_val < 0.05:
            if fdr_val < 0.001:
                fdr_text = "FDR<0.001"
            else:
                fdr_text = f"FDR={fdr_val:.2e}"
            combined_text = f"{p_text}\n{fdr_text}"
            text_color = 'red'
            weight = 'bold'
        else:
            # Only show p-value
            combined_text = p_text
            text_color = 'black' if p_val < 0.05 else 'gray'
            weight = 'bold' if p_val < 0.05 else 'normal'
        
        return combined_text, text_color, weight
    
    # Create the figure
    plt.figure(figsize=(8, 8))
    
    response_order = ['EXT', 'INT', 'NON']
    response_colors = {'EXT': '#2ca02c', 'INT': '#1f77b4', 'NON': '#d62728'}
    # Note: These colors match what you defined in your earlier code
    
    # Create violin plot as base layer
    sns.violinplot(
        x='response_group',
        y='value',
        data=plot_df,
        palette=response_colors,
        order=response_order,
        alpha=0.6,
        inner=None
    )
    
    # Create box plot on top
    sns.boxplot(
        x='response_group',
        y='value',
        data=plot_df,
        order=response_order,
        width=0.4,
        linewidth=1.5,
        fliersize=0,
        boxprops=dict(facecolor='white', alpha=0.8),
        medianprops=dict(color='black', linewidth=2)
    )
    
    # Calculate y-axis positioning for improved layout
    y_max = plot_df['value'].max()
    y_min = plot_df['value'].min()
    y_range = y_max - y_min

    # Start annotations well above the highest violin plot
    annotation_y_start = y_max + y_range * 0.3  # Increased from 0.15 to 0.3
    h = y_range * 0.02  # Height of brackets

    ax = plt.gca()

    # Only show significant comparisons (p < 0.05)
    significant_comparisons = []

    # Check each comparison for significance
    comparisons_info = [
        ('EXT_vs_NON', 0, 2, 'EXT vs NON'),
        ('INT_vs_NON', 1, 2, 'INT vs NON'),
        ('EXT_vs_INT', 0, 1, 'EXT vs INT')
    ]

    for comp_key, x1, x2, label in comparisons_info:
        if comp_key in stats_results:
            p_val = stats_results[comp_key]['p_value']
            if p_val < 0.05:  # Only include significant comparisons
                significant_comparisons.append((comp_key, x1, x2, label))

    # Draw significant comparisons with proper spacing
    for i, (comp_key, x1, x2, label) in enumerate(significant_comparisons):
        # This line needs to be INSIDE the loop where 'i' is defined
        y_pos = annotation_y_start + (i * y_range * 0.15)  # Increased from 0.12 to 0.15

        # Draw bracket
        ax.plot([x1, x1, x2, x2], [y_pos, y_pos+h, y_pos+h, y_pos], lw=1.5, c='black')

        # Get statistics
        p_val = stats_results[comp_key]['p_value']
        fdr_val = stats_results[comp_key].get('p_adjusted', np.nan)
        fold_change = stats_results[comp_key].get('fold_change', np.nan)

        # Format text
        combined_text, text_color, weight = format_p_and_fdr(p_val, fdr_val)

        # Add fold change to the text
        if not pd.isna(fold_change):
            combined_text = f"{fold_change:.2f}-fold\n{combined_text}"

        # Position text
        ax.text((x1+x2)*.5, y_pos+h*1.2, combined_text, ha='center', va='bottom', 
                color=text_color, fontsize=11, fontweight=weight)
    
    # Set y-axis limits with improved spacing
    # For percentage data, never go below 0
    bottom_margin = y_range * 0.25  # Increased bottom margin
    y_bottom = max(0, y_min - bottom_margin)  # Ensure never below 0 for percentage data
    
    # Top margin accounts for annotations
    if significant_comparisons:
        top_margin = annotation_y_start + (len(significant_comparisons) * y_range * 0.12) + y_range * 0.08
    else:
        top_margin = y_max + y_range * 0.15
    
    ax.set_ylim(y_bottom, top_margin)
    
    # Add sample counts below x-axis labels but above xlabel
    for i, resp in enumerate(response_order):
        n = len(plot_df[plot_df['response_group'] == resp])
        # Position sample counts between x-axis labels and xlabel
        ax.annotate(f"n={n}", xy=(i, 0), xytext=(0, -18), 
                   textcoords='offset points', ha='center', va='top',
                   fontsize=10, fontweight='bold')
    
    # Styling
    plt.xlabel('Response Group', fontsize=14)
    plt.ylabel('CD16+ Cells (% of All Cells)', fontsize=14)
    
    # Style matching publication standards
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('CD16_NK_cells_main_figure_improved.png', dpi=600, bbox_inches='tight')
    plt.savefig('CD16_NK_cells_main_figure_improved.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ Improved CD16+ figure saved as CD16_NK_cells_main_figure_improved.png/pdf")
    print(f"✓ Displayed {len(significant_comparisons)} significant comparisons (p < 0.05)")
    
    return stats_results

stats_results = create_main_cd16_figure(flow_df_ready)


# In[32]:


#!/usr/bin/env python
# coding: utf-8

"""
CD25 Plots Generator
Creates individual longitudinal plots for each CD25 measurement type
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

class CD25SeparatePlotsGenerator:
    """
    Generate individual CD25 longitudinal plots for each measurement type
    """
    
    def __init__(self, data_file='CICPT_1558_data (3).xlsx', verbose=True):
        self.verbose = verbose
        self.data_file = data_file
        self.flow_df = None
        self.cd25_data = None
        
        # Define CD25 marker
        self.cd25_parameter = 'CD25-BB515'
        
        # Define timepoint mapping from filename number to biological timepoint
        self.timepoint_mapping = {
            '1': 'B1W1',  # Baseline, Week 1
            '2': 'B2W2',  # Cycle 2, Week 2  
            '3': 'B2W4',  # Cycle 2, Week 4
            '4': 'B4W1'   # Cycle 4, Week 1
        }
        
        # Define expected timepoints based on mapping
        self.expected_timepoints = []
        
        # Define measurement types for CD25
        self.measurement_types = {
            'frequency_parent': {
                'column': '% of parent',
                'name': 'CD25+ Frequency (% of Parent)',
                'ylabel': 'CD25+ cells (% of parent)',
                'filename': 'cd25_frequency_parent',
                'log_transform': False
            },
            'frequency_all': {
                'column': '% of all Cells', 
                'name': 'CD25+ Frequency (% of Total)',
                'ylabel': 'CD25+ cells (% of total)',
                'filename': 'cd25_frequency_total',
                'log_transform': False
            },
            'mfi_median': {
                'column': 'Median',
                'name': 'CD25 Expression (MFI Median)',
                'ylabel': 'CD25 MFI (Median)',
                'filename': 'cd25_mfi_median',
                'log_transform': True
            },
            'mfi_geometric': {
                'column': 'Geometric Mean',
                'name': 'CD25 Expression (Geometric Mean MFI)',
                'ylabel': 'CD25 MFI (Geometric Mean)',
                'filename': 'cd25_mfi_geometric',
                'log_transform': True
            }
        }
        
        if self.verbose:
            print("CD25 SEPARATE PLOTS GENERATOR")
            print("="*60)
        
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load and prepare flow data with response group and timepoint mapping"""
        
        if self.verbose:
            print("Loading and preparing data...")
        
        try:
            # Load all necessary sheets
            sample_info_df = pd.read_excel(self.data_file, sheet_name='sampleID', header=None)
            flow_df = pd.read_excel(self.data_file, sheet_name=2)  # Flow data sheet
            response_df = pd.read_excel(self.data_file, sheet_name='group', header=None)
            
            if self.verbose:
                print(f"✓ Flow data: {flow_df.shape}")
                print(f"✓ Sample info: {sample_info_df.shape}")
                print(f"✓ Response groups: {response_df.shape}")
                
        except Exception as e:
            print(f" Error loading data: {e}")
            return
        
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
        
        # Create Study ID to response mapping
        studyid_to_response = {}
        for i in range(len(sample_info_df)):
            mrn = sample_info_df.iloc[i, 0]  # Column A
            study_id = sample_info_df.iloc[i, 1]  # Column B
            
            if pd.isna(mrn) or pd.isna(study_id):
                continue
            
            # Try different MRN formats
            mrn_variants = [mrn]
            if str(mrn).replace('.','').replace('-','').isdigit():
                mrn_variants.extend([int(float(mrn)), str(int(float(mrn)))])
            mrn_variants.append(str(mrn))
            mrn_variants = [m for m in set(mrn_variants) if m is not None]
            
            response = None
            for mrn_variant in mrn_variants:
                response = mrn_to_response.get(mrn_variant)
                if response:
                    break
            
            if response and study_id:
                studyid_to_response[study_id] = response
        
        # Add response groups and timepoints to flow data
        flow_df['response_group'] = None
        flow_df['patient_id'] = None
        flow_df['timepoint'] = None
        
        mapped_samples = 0
        for idx, row in flow_df.iterrows():
            filename = str(row.get('Filename', ''))
            
            # Extract patient number from filename
            pt_match = re.search(r'PT\s*(\d+)', filename)
            
            # Extract timepoint from filename
            # Look for patterns like PT 23-1, PT 24-2, etc.
            # The timepoint is the number after the dash
            timepoint_match = re.search(r'PT\s*\d+-(\d+)', filename)
            
            if pt_match:
                patient_num = pt_match.group(1).zfill(3)  # Pad to 3 digits
                study_id = f"01-{patient_num}"
                
                response = studyid_to_response.get(study_id)
                if response:
                    flow_df.at[idx, 'response_group'] = response
                    flow_df.at[idx, 'patient_id'] = study_id
                    mapped_samples += 1
            
            # Add timepoint information
            if timepoint_match:
                timepoint_num = timepoint_match.group(1)
                # Map filename timepoint number to biological timepoint
                timepoint = self.timepoint_mapping.get(timepoint_num, f"T{timepoint_num}")
                flow_df.at[idx, 'timepoint'] = timepoint
        
        self.flow_df = flow_df
        
        if self.verbose:
            print(f"✓ Mapped samples: {mapped_samples}")
            
            # Show response group distribution
            response_counts = flow_df['response_group'].value_counts()
            print("Response group distribution:")
            for response, count in response_counts.items():
                print(f"  {response}: {count}")
            
            # Show timepoint distribution
            timepoint_counts = flow_df['timepoint'].value_counts()
            print("Timepoint distribution:")
            for timepoint, count in timepoint_counts.items():
                print(f"  {timepoint}: {count}")
            
            # Update expected timepoints based on actual data
            actual_timepoints = [tp for tp in flow_df['timepoint'].unique() if pd.notna(tp)]
            if actual_timepoints:
                # Sort timepoints in biological order (B1W1, B2W2, B2W4, B4W1)
                timepoint_order = ['B1W1', 'B2W2', 'B2W4', 'B4W1']
                self.expected_timepoints = [tp for tp in timepoint_order if tp in actual_timepoints]
                print(f"✓ Found timepoints: {self.expected_timepoints}")
    
    def extract_cd25_data(self):
        """Extract CD25 data across timepoints and response groups"""
        
        if self.flow_df is None:
            print(" No flow data available. Run load_and_prepare_data first.")
            return
        
        if self.verbose:
            print("\nExtracting CD25 data...")
        
        # Filter for CD25 parameter only
        cd25_df = self.flow_df[self.flow_df['Parameter'] == self.cd25_parameter].copy()
        
        if len(cd25_df) == 0:
            print(f" No CD25 data found for parameter: {self.cd25_parameter}")
            return
        
        # Filter for samples with both response group and timepoint
        cd25_df = cd25_df[cd25_df['response_group'].notna() & cd25_df['timepoint'].notna()]
        
        if self.verbose:
            print(f"✓ CD25 samples with complete data: {len(cd25_df)}")
            
            # Show data availability by timepoint and response group
            pivot_counts = cd25_df.groupby(['timepoint', 'response_group']).size().unstack(fill_value=0)
            print("\nSample counts by timepoint and response group:")
            print(pivot_counts)
        
        self.cd25_data = cd25_df
        return cd25_df
    
    def create_single_plot(self, measurement_type, figsize=(10, 6), save_path=None, dpi=300):
        """Create a single longitudinal plot for one measurement type"""
        
        if self.cd25_data is None:
            self.extract_cd25_data()
        
        if self.cd25_data is None or len(self.cd25_data) == 0:
            print(" No CD25 data available for plotting")
            return None
        
        if measurement_type not in self.measurement_types:
            print(f" Invalid measurement type: {measurement_type}")
            print(f"Available types: {list(self.measurement_types.keys())}")
            return None
        
        measurement_info = self.measurement_types[measurement_type]
        measurement_col = measurement_info['column']
        
        # Check if measurement column is available
        if measurement_col not in self.cd25_data.columns:
            print(f" {measurement_col} not available in data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for plotting
        plot_data = []
        
        for _, row in self.cd25_data.iterrows():
            value = pd.to_numeric(row[measurement_col], errors='coerce')
            
            if pd.notna(value):
                # Apply log transformation if specified
                if measurement_info['log_transform']:
                    value = np.log2(value + 1)
                
                plot_data.append({
                    'timepoint': row['timepoint'],
                    'response_group': row['response_group'],
                    'patient_id': row['patient_id'],
                    'value': value
                })
        
        if len(plot_data) == 0:
            print(f" No valid data for {measurement_col}")
            plt.close(fig)
            return None
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        self._plot_longitudinal_data(ax, plot_df, measurement_info)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            try:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                if self.verbose:
                    print(f"✓ Figure saved to: {save_path}")
            except Exception as e:
                print(f" Error saving figure: {e}")
        
        return fig
    
    def create_all_separate_plots(self, figsize=(10, 6), save_directory=None, 
                                  file_format='png', dpi=600):
        """Create separate plots for all measurement types"""
        
        if self.cd25_data is None:
            self.extract_cd25_data()
        
        if self.cd25_data is None:
            return {}
        
        figures = {}
        
        for measurement_type, measurement_info in self.measurement_types.items():
            
            if self.verbose:
                print(f"\nCreating plot for: {measurement_info['name']}")
            
            # Determine save path
            save_path = None
            if save_directory:
                filename = f"{measurement_info['filename']}.{file_format}"
                save_path = f"{save_directory}/{filename}"
            
            # Create the plot
            fig = self.create_single_plot(measurement_type, figsize=figsize, 
                                        save_path=save_path, dpi=dpi)
            
            if fig is not None:
                figures[measurement_type] = fig
                if self.verbose:
                    print(f"✓ Created plot for {measurement_type}")
            else:
                if self.verbose:
                    print(f" Failed to create plot for {measurement_type}")
        
        return figures
    
    def _plot_longitudinal_data(self, ax, plot_df, measurement_info):
        """Helper function to create individual longitudinal plots"""
        
        # Define colors for response groups
        colors = {'EXT': 'green', 'INT': 'blue', 'NON': 'red'}
        
        # Order timepoints properly
        timepoint_order = [tp for tp in self.expected_timepoints if tp in plot_df['timepoint'].unique()]
        
        if len(timepoint_order) == 0:
            ax.text(0.5, 0.5, 'No Valid Timepoints', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            return
        
        # Plot individual patient trajectories (thin lines)
        for response_group in ['EXT', 'INT', 'NON']:
            group_data = plot_df[plot_df['response_group'] == response_group]
            
            if len(group_data) == 0:
                continue
            
            # Plot individual patient trajectories
            for patient_id in group_data['patient_id'].unique():
                patient_data = group_data[group_data['patient_id'] == patient_id]
                patient_data = patient_data.sort_values('timepoint')
                
                # Only plot if patient has multiple timepoints
                if len(patient_data) > 1:
                    timepoint_numeric = [timepoint_order.index(tp) for tp in patient_data['timepoint'] 
                                       if tp in timepoint_order]
                    values = [patient_data[patient_data['timepoint'] == timepoint_order[i]]['value'].iloc[0] 
                             for i in timepoint_numeric]
                    
                    ax.plot(timepoint_numeric, values, color=colors[response_group], 
                           alpha=0.3, linewidth=0.8)
        
        # Plot group means with error bars
        for response_group in ['EXT', 'INT', 'NON']:
            group_data = plot_df[plot_df['response_group'] == response_group]
            
            if len(group_data) == 0:
                continue
            
            means = []
            stds = []
            timepoint_positions = []
            
            for i, timepoint in enumerate(timepoint_order):
                tp_data = group_data[group_data['timepoint'] == timepoint]['value']
                
                if len(tp_data) > 0:
                    means.append(tp_data.mean())
                    stds.append(tp_data.std())
                    timepoint_positions.append(i)
            
            if len(means) > 0:
                # Plot mean line with error bars
                ax.errorbar(timepoint_positions, means, yerr=stds, 
                           color=colors[response_group], linewidth=3, 
                           marker='o', markersize=8, capsize=5, capthick=2,
                           label=f'{response_group} (n={len(group_data["patient_id"].unique())})')
        
        # Customize plot
        ax.set_xticks(range(len(timepoint_order)))
        ax.set_xticklabels(timepoint_order)
        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel(measurement_info['ylabel'], fontsize=12)
        ax.set_title(measurement_info['name'], fontsize=13)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add sample size annotations
        y_range = ax.get_ylim()
        y_pos = y_range[0] + 0.05 * (y_range[1] - y_range[0])
        
        for i, timepoint in enumerate(timepoint_order):
            tp_data = plot_df[plot_df['timepoint'] == timepoint]
            n_samples = len(tp_data)
            ax.text(i, y_pos, f'n={n_samples}', ha='center', va='top', 
                   fontsize=10, alpha=0.7, fontweight='bold')

def run_separate_cd25_plots(data_file='CICPT_1558_data (3).xlsx', 
                           save_directory=None, file_format='png', 
                           figsize=(10, 6), dpi=300):
    
    # Initialize generator
    generator = CD25SeparatePlotsGenerator(data_file, verbose=True)
    
    # Create all separate plots
    figures = generator.create_all_separate_plots(figsize=figsize, 
                                                 save_directory=save_directory,
                                                 file_format=file_format, 
                                                 dpi=dpi)
    
    if len(figures) > 0:
        print(f"\n✓ Created {len(figures)} separate plots")
        if save_directory:
            print(f"✓ Plots saved to: {save_directory}/")
    else:
        print(" No plots were created")
    
    return generator, figures

def create_individual_plot(measurement_type, data_file='CICPT_1558_data (3).xlsx',
                          save_path=None, figsize=(10, 6), dpi=300):
    """
    Create a single plot for one measurement type
    
    Parameters:
    - measurement_type: One of 'frequency_parent', 'frequency_all', 'mfi_median', 'mfi_geometric'
    - data_file: Path to Excel file with flow data
    - save_path: Path to save the figure (optional)
    - figsize: Size of the plot (width, height)
    - dpi: DPI for saved figure
    
    Returns:
    - generator: CD25SeparatePlotsGenerator instance
    - fig: matplotlib figure
    """
    
    # Initialize generator
    generator = CD25SeparatePlotsGenerator(data_file, verbose=True)
    
    # Create single plot
    fig = generator.create_single_plot(measurement_type, figsize=figsize,
                                      save_path=save_path, dpi=dpi)
    
    return generator, fig

generator, figures = run_separate_cd25_plots(
    'CICPT_1558_data (3).xlsx',
    save_directory='cd25_plots',
    file_format='png')


# In[33]:


# Flow Cytometry Analysis: Pembrolizumab + HD IL-2 in mRCC
# Analysis of PD-1 expression changes across treatment timepoints

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=== FLOW CYTOMETRY DATA ANALYSIS ===")
print("Study: Sequential Pembrolizumab + HD IL-2 in mRCC")
print("Analysis: PD-1 expression changes across treatment timeline")
print("Timepoints: B1W1 (baseline) → B2W2 (post-pembro) → B2W4 (post-IL-2) → B4W1 (recovery)")
print("="*60)

# ========================
# 1. DATA LOADING & SETUP
# ========================

print("\n1. LOADING DATA...")

# Load the Excel file
file_path = "18536 PD-1 flow data pivot 250611.xlsx"

try:
    # Read both sheets
    data_sheet = pd.read_excel(file_path, sheet_name='Data')
    lookup_sheet = pd.read_excel(file_path, sheet_name='SmplLookup')
    
    print(f"✓ Data sheet loaded: {data_sheet.shape[0]} rows, {data_sheet.shape[1]} columns")
    print(f"✓ Lookup sheet loaded: {lookup_sheet.shape[0]} rows, {lookup_sheet.shape[1]} columns")
    
    # Display data structure
    print("\nData sheet columns:", list(data_sheet.columns))
    print("Lookup sheet columns:", list(lookup_sheet.columns))
    
except FileNotFoundError:
    print(" File not found. Please ensure '18536 PD-1 flow data pivot 250611.xlsx' is in the working directory")


# ========================
# 2. DATA PREPROCESSING
# ========================

print("\n2. DATA PREPROCESSING...")

# Merge data with lookup information
merged_data = data_sheet.merge(lookup_sheet, left_on='Filename', right_on='Flow ID', how='left')

print(f"✓ Data merged: {merged_data.shape[0]} rows")
print(f"  - Matched samples: {merged_data['Subject'].notna().sum()}")
print(f"  - Unmatched samples: {merged_data['Subject'].isna().sum()}")

# Clean column names
merged_data.columns = [col.strip() for col in merged_data.columns]
merged_data['Percent_Parent'] = pd.to_numeric(merged_data['% of parent'], errors='coerce')
merged_data['Median_FI'] = pd.to_numeric(merged_data['Median'], errors='coerce')
merged_data['Gate'] = merged_data['Gate name']  # Standardize gate column name

print(f"Data type cleaning:")
print(f"  - % of parent: {merged_data['Percent_Parent'].isna().sum()} non-numeric values converted to NaN")
print(f"  - Median: {merged_data['Median_FI'].isna().sum()} non-numeric values converted to NaN")

# Separate experimental samples from controls
experimental_data = merged_data[~merged_data['Subject'].isin(['FMO Ctrl', 'PBMC Ctrl'])].copy()
control_data = merged_data[merged_data['Subject'].isin(['FMO Ctrl', 'PBMC Ctrl'])].copy()

print(f"✓ Experimental samples: {len(experimental_data)}")
print(f"✓ Control samples: {len(control_data)}")

# Check timepoint distribution
print("\nTimepoint distribution:")
timepoint_counts = experimental_data['Timepoint'].value_counts().sort_index()
print(timepoint_counts)

# Check gate distribution
print("\nGate distribution:")
gate_counts = experimental_data['Gate'].value_counts()
print(gate_counts)

# ========================
# 3. DATA VALIDATION
# ========================

print("\n3. DATA VALIDATION...")

# Check for missing values
missing_check = experimental_data.isnull().sum()
print("Missing values per column:")
print(missing_check[missing_check > 0])

# Check for outliers in key measurements
def identify_outliers(data, column, method='iqr'):
    """Identify outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Check outliers for each gate
print("\nOutlier analysis:")
for gate in experimental_data['Gate'].unique():
    gate_data = experimental_data[experimental_data['Gate'] == gate]
    
    pct_outliers = identify_outliers(gate_data, 'Percent_Parent')
    fi_outliers = identify_outliers(gate_data, 'Median_FI')
    
    print(f"{gate}:")
    print(f"  - % Parent outliers: {len(pct_outliers)} ({len(pct_outliers)/len(gate_data)*100:.1f}%)")
    print(f"  - Median FI outliers: {len(fi_outliers)} ({len(fi_outliers)/len(gate_data)*100:.1f}%)")

# Validate patient longitudinal data
subjects = experimental_data['Subject'].unique()
print(f"\nLongitudinal data validation:")
print(f"Total subjects: {len(subjects)}")

# Check which subjects have complete timepoint data
complete_subjects = []
incomplete_subjects = []

expected_timepoints = ['B1W1', 'B2W2', 'B2W4', 'B4W1']

for subject in subjects:
    subject_data = experimental_data[experimental_data['Subject'] == subject]
    subject_timepoints = set(subject_data['Timepoint'].unique())
    
    if set(expected_timepoints).issubset(subject_timepoints):
        complete_subjects.append(subject)
    else:
        incomplete_subjects.append(subject)
        missing_tp = set(expected_timepoints) - subject_timepoints
        print(f"  {subject}: Missing {missing_tp}")

print(f"✓ Complete longitudinal data: {len(complete_subjects)} subjects")
print(f" Incomplete data: {len(incomplete_subjects)} subjects")

# ========================
# 4. CONTROL ANALYSIS
# ========================

print("\n4. CONTROL ANALYSIS...")

if len(control_data) > 0:
    print("Control sample summary:")
    control_summary = control_data.groupby(['Subject', 'Gate']).agg({
        'Percent_Parent': ['mean', 'std'],
        'Median_FI': ['mean', 'std']
    }).round(3)
    
    print(control_summary)
    
    # Flag any controls with unexpectedly high values
    print("\nControl validation (flagging high values):")
    for gate in control_data['Gate'].unique():
        gate_controls = control_data[control_data['Gate'] == gate]
        high_pct = gate_controls[gate_controls['Percent_Parent'] > 5.0]  # Threshold for high percentage
        
        if len(high_pct) > 0:
            print(f" High % parent in {gate} controls:")
            for _, row in high_pct.iterrows():
                print(f"  {row['Subject']}: {row['Percent_Parent']:.2f}%")

# ========================
# 5. DESCRIPTIVE STATISTICS
# ========================

print("\n5. DESCRIPTIVE STATISTICS...")

# Calculate summary statistics by timepoint and gate
summary_stats = experimental_data.groupby(['Timepoint', 'Gate']).agg({
    'Percent_Parent': ['count', 'mean', 'std', 'median', 'min', 'max'],
    'Median_FI': ['mean', 'std', 'median', 'min', 'max']
}).round(3)

print("Summary statistics by timepoint and gate:")
print(summary_stats)

# Create visualization function
def create_summary_plot():
    """Create summary visualization of the data"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Flow Cytometry Data Overview: PD-1 Expression Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: % Parent by timepoint (all gates)
    ax1 = axes[0, 0]
    experimental_data.boxplot(column='Percent_Parent', by='Timepoint', ax=ax1)
    ax1.set_title('% Parent by Timepoint (All Gates)')
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('% of Parent')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Median FI by timepoint (all gates)
    ax2 = axes[0, 1]
    experimental_data.boxplot(column='Median_FI', by='Timepoint', ax=ax2)
    ax2.set_title('Median Fluorescence Intensity by Timepoint')
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel('Median FI')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: % Parent by gate
    ax3 = axes[1, 0]
    experimental_data.boxplot(column='Percent_Parent', by='Gate', ax=ax3)
    ax3.set_title('% Parent by Gate (All Timepoints)')
    ax3.set_xlabel('Gate')
    ax3.set_ylabel('% of Parent')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Sample count by timepoint
    ax4 = axes[1, 1]
    timepoint_counts.plot(kind='bar', ax=ax4, color='skyblue')
    ax4.set_title('Sample Count by Timepoint')
    ax4.set_xlabel('Timepoint')
    ax4.set_ylabel('Number of Samples')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

# Create and display the overview plot
overview_fig = create_summary_plot()
plt.show()

print("\n" + "="*60)
print("DATA LOADING AND VALIDATION COMPLETE")
print("="*60)
print(f"✓ Total experimental samples: {len(experimental_data)}")
print(f"✓ Subjects with complete data: {len(complete_subjects)}")
print(f"✓ Gates analyzed: {list(experimental_data['Gate'].unique())}")
print(f"✓ Timepoints: {list(experimental_data['Timepoint'].unique())}")


# In[37]:


# ========================
# MISSING DATA ANALYSIS & HANDLING
# ========================

print("\n=== MISSING DATA ANALYSIS ===")
print("Addressing longitudinal dropout in clinical trial data")

# ========================
# 1. DROPOUT PATTERN ANALYSIS
# ========================

print("\n1. DROPOUT PATTERN ANALYSIS...")

def analyze_dropout_patterns():
    """Analyze patterns of missing data across timepoints"""
    subjects = experimental_data['Subject'].unique()
    expected_timepoints = ['B1W1', 'B2W2', 'B2W4', 'B4W1']
    
    dropout_analysis = []
    
    for subject in subjects:
        subject_data = experimental_data[experimental_data['Subject'] == subject]
        available_timepoints = set(subject_data['Timepoint'].unique())
        
        # Determine dropout pattern
        pattern = []
        last_available = None
        
        for tp in expected_timepoints:
            if tp in available_timepoints:
                pattern.append('Present')
                last_available = tp
            else:
                pattern.append('Missing')
        
        # Classify dropout type
        if all(p == 'Present' for p in pattern):
            dropout_type = 'Complete'
        elif pattern[0] == 'Missing':
            dropout_type = 'No_Baseline'
        else:
            # Find last present timepoint
            last_idx = max([i for i, p in enumerate(pattern) if p == 'Present'])
            if last_idx == 0:  # Only baseline
                dropout_type = 'Early_Dropout'
            elif last_idx == 1:  # Through B2W2
                dropout_type = 'Post_Pembrolizumab_Dropout'
            elif last_idx == 2:  # Through B2W4
                dropout_type = 'Post_IL2_Dropout'
            else:
                dropout_type = 'Late_Dropout'
        
        dropout_analysis.append({
            'Subject': subject,
            'B1W1': pattern[0],
            'B2W2': pattern[1], 
            'B2W4': pattern[2],
            'B4W1': pattern[3],
            'Dropout_Type': dropout_type,
            'Last_Available': last_available,
            'N_Timepoints': sum(1 for p in pattern if p == 'Present')
        })
    
    return pd.DataFrame(dropout_analysis)

dropout_df = analyze_dropout_patterns()

print(f"Total subjects: {len(dropout_df)}")
print(f"Complete data: {len(dropout_df[dropout_df['Dropout_Type'] == 'Complete'])} ({len(dropout_df[dropout_df['Dropout_Type'] == 'Complete'])/len(dropout_df)*100:.1f}%)")

print("\nDropout patterns:")
dropout_counts = dropout_df['Dropout_Type'].value_counts()
for dropout_type, count in dropout_counts.items():
    print(f"  {dropout_type}: {count} ({count/len(dropout_df)*100:.1f}%)")

print("\nTimepoint availability:")
for tp in ['B1W1', 'B2W2', 'B2W4', 'B4W1']:
    available = (dropout_df[tp] == 'Present').sum()
    print(f"  {tp}: {available}/{len(dropout_df)} ({available/len(dropout_df)*100:.1f}%)")

# ========================
# 2. AVAILABLE CASE ANALYSIS
# ========================

print("\n2. AVAILABLE CASE ANALYSIS...")
print("Using all available data at each timepoint (not just complete cases)")

def available_case_analysis():
    """Perform analysis using all available data at each timepoint"""
    
    # Calculate summary statistics using all available data
    timepoint_summary = []
    
    for gate in experimental_data['Gate'].unique():
        gate_data = experimental_data[experimental_data['Gate'] == gate]
        
        for timepoint in ['B1W1', 'B2W2', 'B2W4', 'B4W1']:
            tp_data = gate_data[gate_data['Timepoint'] == timepoint]
            
            if len(tp_data) > 0:
                # Handle duplicates by taking median
                tp_clean = tp_data.groupby('Subject').agg({
                    'Percent_Parent': 'median',
                    'Median_FI': 'median'
                }).reset_index()
                
                timepoint_summary.append({
                    'Gate': gate,
                    'Timepoint': timepoint,
                    'N': len(tp_clean),
                    'Percent_Parent_Median': tp_clean['Percent_Parent'].median(),
                    'Percent_Parent_IQR_25': tp_clean['Percent_Parent'].quantile(0.25),
                    'Percent_Parent_IQR_75': tp_clean['Percent_Parent'].quantile(0.75),
                    'Percent_Parent_Mean': tp_clean['Percent_Parent'].mean(),
                    'Percent_Parent_SD': tp_clean['Percent_Parent'].std(),
                    'Median_FI_Mean': tp_clean['Median_FI'].mean(),
                    'Median_FI_SD': tp_clean['Median_FI'].std()
                })
    
    return pd.DataFrame(timepoint_summary)

available_case_results = available_case_analysis()

print("Available case sample sizes by timepoint and gate:")
sample_sizes = available_case_results.pivot(index='Gate', columns='Timepoint', values='N')
print(sample_sizes)

# ========================
# 3. COMPARISON: COMPLETE CASE vs AVAILABLE CASE
# ========================

print("\n3. COMPLETE CASE vs AVAILABLE CASE COMPARISON...")

def compare_analysis_approaches():
    """Compare results between complete case and available case analysis"""
    
    # Focus on main gate for comparison
    main_gate = 'Live+/PD-1+'
    if main_gate not in experimental_data['Gate'].unique():
        main_gate = experimental_data['Gate'].unique()[0]
    
    print(f"Comparing approaches for {main_gate}:")
    
    # Available case results
    avail_results = available_case_results[available_case_results['Gate'] == main_gate]
    
    # Complete case results (using previous complete_data_agg if available)
    if 'complete_data_agg' in locals():
        complete_results = complete_data_agg[complete_data_agg['Gate'] == main_gate].groupby('Timepoint').agg({
            'Percent_Parent': ['count', 'median', 'mean', 'std']
        }).round(3)
    
    print("\nSample sizes:")
    print("Available case approach:")
    for _, row in avail_results.iterrows():
        print(f"  {row['Timepoint']}: N={row['N']}")
    
    if 'complete_data_agg' in locals():
        print("Complete case approach:")
        for tp in ['B1W1', 'B2W2', 'B2W4', 'B4W1']:
            if tp in complete_results.index:
                n = complete_results.loc[tp, ('Percent_Parent', 'count')]
                print(f"  {tp}: N={n}")
    
    print("\nMedian % Parent values:")
    print("Available case:")
    for _, row in avail_results.iterrows():
        print(f"  {row['Timepoint']}: {row['Percent_Parent_Median']:.2f}")
    
    return avail_results

comparison_results = compare_analysis_approaches()

# ========================
# 4. MIXED-EFFECTS MODEL ANALYSIS
# ========================

print("\n4. MIXED-EFFECTS MODEL ANALYSIS...")
print("Using linear mixed-effects models to handle missing data properly")

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    
    def run_mixed_effects_analysis():
        """Run mixed-effects models for each gate"""
        
        mixed_results = {}
        
        for gate in experimental_data['Gate'].unique():
            gate_data = experimental_data[experimental_data['Gate'] == gate].copy()
            
            # Handle duplicates
            gate_clean = gate_data.groupby(['Subject', 'Timepoint']).agg({
                'Percent_Parent': 'median',
                'Median_FI': 'median'
            }).reset_index()
            
            # Add timepoint as numeric for trend analysis
            timepoint_map = {'B1W1': 0, 'B2W2': 1, 'B2W4': 2, 'B4W1': 3}
            gate_clean['Timepoint_num'] = gate_clean['Timepoint'].map(timepoint_map)
            
            # Remove rows with missing data
            gate_clean = gate_clean.dropna(subset=['Percent_Parent', 'Timepoint_num'])
            
            if len(gate_clean) >= 10:  # Need sufficient data for mixed model
                try:
                    # Fit mixed-effects model with random intercept for subject
                    model = mixedlm("Percent_Parent ~ Timepoint_num", 
                                  gate_clean, 
                                  groups=gate_clean["Subject"])
                    result = model.fit()
                    
                    mixed_results[gate] = {
                        'model': result,
                        'slope': result.params['Timepoint_num'],
                        'slope_pvalue': result.pvalues['Timepoint_num'],
                        'n_observations': len(gate_clean),
                        'n_subjects': gate_clean['Subject'].nunique()
                    }
                    
                    print(f"\n{gate}:")
                    print(f"  Slope (change per timepoint): {result.params['Timepoint_num']:.3f}")
                    print(f"  P-value: {result.pvalues['Timepoint_num']:.4f}")
                    print(f"  N observations: {len(gate_clean)}")
                    print(f"  N subjects: {gate_clean['Subject'].nunique()}")
                    
                except Exception as e:
                    print(f"  Mixed model failed for {gate}: {str(e)}")
            else:
                print(f"  Insufficient data for {gate} mixed model (N={len(gate_clean)})")
        
        return mixed_results
    
    mixed_effects_results = run_mixed_effects_analysis()
    
except ImportError:
    print("statsmodels not available - install with 'pip install statsmodels' for mixed-effects models")
    mixed_effects_results = {}

# ========================
# 5. SENSITIVITY ANALYSIS
# ========================

print("\n5. SENSITIVITY ANALYSIS...")

def sensitivity_analysis():
    """Compare different approaches to missing data"""
    
    main_gate = 'Live+/PD-1+'
    if main_gate not in experimental_data['Gate'].unique():
        main_gate = experimental_data['Gate'].unique()[0]
    
    gate_data = experimental_data[experimental_data['Gate'] == main_gate]
    
    # Clean duplicates
    gate_clean = gate_data.groupby(['Subject', 'Timepoint']).agg({
        'Percent_Parent': 'median'
    }).reset_index()
    
    approaches = {}
    
    # 1. Available case analysis
    avail_summary = gate_clean.groupby('Timepoint')['Percent_Parent'].agg(['count', 'median', 'mean']).round(3)
    approaches['Available_Case'] = avail_summary
    
    # 2. Complete case analysis  
    if 'complete_subjects' in locals():
        complete_gate = gate_clean[gate_clean['Subject'].isin(complete_subjects)]
        complete_summary = complete_gate.groupby('Timepoint')['Percent_Parent'].agg(['count', 'median', 'mean']).round(3)
        approaches['Complete_Case'] = complete_summary
    
    # 3. Last observation carried forward (LOCF)
    pivot_data = gate_clean.pivot(index='Subject', columns='Timepoint', values='Percent_Parent')
    locf_data = pivot_data.fillna(method='ffill', axis=1)  # Forward fill
    locf_summary = locf_data.mean().to_frame('mean')
    locf_summary['count'] = locf_data.count()
    locf_summary['median'] = locf_data.median()
    approaches['LOCF'] = locf_summary[['count', 'median', 'mean']].round(3)
    
    print(f"Sensitivity analysis for {main_gate} - % Parent values:")
    print("\nApproach comparison (Median values):")
    
    for timepoint in ['B1W1', 'B2W2', 'B2W4', 'B4W1']:
        print(f"\n{timepoint}:")
        for approach, data in approaches.items():
            if timepoint in data.index:
                median_val = data.loc[timepoint, 'median']
                n_val = data.loc[timepoint, 'count'] 
                print(f"  {approach}: {median_val:.2f} (N={n_val})")
    
    return approaches

sensitivity_results = sensitivity_analysis()


# ========================
# 6. UPDATED ANALYSIS USING ALL AVAILABLE DATA
# ========================

print("\n7. UPDATED TRAJECTORY ANALYSIS (ALL AVAILABLE DATA)...")

def create_available_case_plots():
    """Create plots using all available data"""
    gates = experimental_data['Gate'].unique()
    n_gates = len(gates)
    
    fig, axes = plt.subplots(n_gates, 2, figsize=(16, 4*n_gates))
    if n_gates == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('PD-1 Expression Trajectories: All Available Data\n(Available Case Analysis)', 
                 fontsize=16, fontweight='bold')
    
    for i, gate in enumerate(gates):
        gate_data = experimental_data[experimental_data['Gate'] == gate]
        
        # Clean duplicates
        gate_clean = gate_data.groupby(['Subject', 'Timepoint']).agg({
            'Percent_Parent': 'median',
            'Median_FI': 'median'
        }).reset_index()
        
        # Add timepoint numbers
        timepoint_map = {'B1W1': 0, 'B2W2': 1, 'B2W4': 2, 'B4W1': 3}
        gate_clean['Timepoint_num'] = gate_clean['Timepoint'].map(timepoint_map)
        
        # Plot % Parent trajectories  
        ax1 = axes[i, 0]
        
        # Individual patient trajectories (show all available data)
        subjects = gate_clean['Subject'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))
        
        for j, subject in enumerate(subjects):
            subject_data = gate_clean[gate_clean['Subject'] == subject].sort_values('Timepoint_num')
            if len(subject_data) >= 2:  # Need at least 2 points to draw line
                ax1.plot(subject_data['Timepoint_num'], subject_data['Percent_Parent'], 
                        'o-', alpha=0.4, color=colors[j], linewidth=1, markersize=4)
        
        # Mean trajectory with error bars (using all available data at each timepoint)
        mean_data = gate_clean.groupby('Timepoint_num')['Percent_Parent'].agg(['mean', 'sem', 'count']).reset_index()
        ax1.errorbar(mean_data['Timepoint_num'], mean_data['mean'], yerr=mean_data['sem'],
                    fmt='ko-', linewidth=3, markersize=8, capsize=5, capthick=2, label='Mean ± SEM')
        
        # Add sample sizes
        for _, row in mean_data.iterrows():
            ax1.text(row['Timepoint_num'], row['mean'] + row['sem'] + 0.5, 
                    f"n={int(row['count'])}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax1.set_title(f'{gate} - % Parent (Available Case Analysis)')
        ax1.set_xlabel('Treatment Phase')
        ax1.set_ylabel('% of Parent')
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(['Baseline\n(B1W1)', 'Post-Pembro\n(B2W2)', 
                            'Post-IL-2\n(B2W4)', 'Recovery\n(B4W1)'])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot Median FI trajectories
        ax2 = axes[i, 1]
        
        # Individual patient trajectories
        for j, subject in enumerate(subjects):
            subject_data = gate_clean[gate_clean['Subject'] == subject].sort_values('Timepoint_num')
            if len(subject_data) >= 2:
                ax2.plot(subject_data['Timepoint_num'], subject_data['Median_FI'], 
                        'o-', alpha=0.4, color=colors[j], linewidth=1, markersize=4)
        
        # Mean trajectory with error bars
        mean_data_fi = gate_clean.groupby('Timepoint_num')['Median_FI'].agg(['mean', 'sem', 'count']).reset_index()
        ax2.errorbar(mean_data_fi['Timepoint_num'], mean_data_fi['mean'], yerr=mean_data_fi['sem'],
                    fmt='ko-', linewidth=3, markersize=8, capsize=5, capthick=2, label='Mean ± SEM')
        
        # Add sample sizes
        for _, row in mean_data_fi.iterrows():
            ax2.text(row['Timepoint_num'], row['mean'] + row['sem'] + 20, 
                    f"n={int(row['count'])}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax2.set_title(f'{gate} - Median FI (Available Case Analysis)')
        ax2.set_xlabel('Treatment Phase')
        ax2.set_ylabel('Median FI')
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['Baseline\n(B1W1)', 'Post-Pembro\n(B2W2)', 
                            'Post-IL-2\n(B2W4)', 'Recovery\n(B4W1)'])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    return fig

# Create available case plots
available_case_fig = create_available_case_plots()
plt.show()

print(f"\n=== MISSING DATA ANALYSIS COMPLETE ===")
print(f"✓ Analyzed dropout patterns for {len(dropout_df)} subjects")
print(f"✓ Implemented available case analysis using all data")
print(f"✓ Compared complete case vs available case approaches") 
print(f"✓ Provided sensitivity analysis and manuscript recommendations")


# In[43]:


# ========================
# RESPONSE DATA PROCESSOR AND ANALYSIS
# ========================

print("=== LOADING CLINICAL RESPONSE DATA ===")

# ========================
# 1. LOAD RESPONSE DATA FROM EXCEL
# ========================

def load_response_data():
    """Load patient response data from Excel file"""
    
    try:
        # Load the response correlation file
        response_file = "ID response correlation.xlsx"
        response_raw = pd.read_excel(response_file)
        
        print(f"✓ Response data loaded: {len(response_raw)} patients")
        print("Raw data preview:")
        print(response_raw.head(10))
        
        # Clean column names
        response_raw.columns = [col.strip() for col in response_raw.columns]
        
        # Display column names and data
        print(f"\nColumns found: {list(response_raw.columns)}")
        
        return response_raw
        
    except FileNotFoundError:
        print(" File 'ID response correlation.xlsx' not found")

        
# ========================
# 2. MAP PATIENT IDS TO FLOW CYTOMETRY FORMAT
# ========================

def create_response_mapping(response_raw):
    """Create mapping between patient IDs and response categories"""
    
    print("\n=== CREATING PATIENT ID MAPPING ===")
    
    # Check what subject IDs exist in the flow cytometry data
    flow_subjects = experimental_data['Subject'].unique()
    print(f"Flow cytometry subjects found: {len(flow_subjects)}")
    print("Sample flow subjects:", flow_subjects[:10])
    
    # Check format of response data IDs
    response_ids = response_raw.iloc[:, 0].astype(str)  # First column should be patient IDs
    response_categories = response_raw.iloc[:, 1]  # Second column should be response categories
    
    print(f"\nResponse data IDs found: {len(response_ids)}")
    print("Sample response IDs:", response_ids.head().tolist())
    
    # Try different mapping strategies
    mapping_strategies = []
    
    # Strategy 1: Direct match (unlikely but check)
    direct_matches = set(response_ids) & set(flow_subjects)
    if direct_matches:
        print(f"✓ Direct matches found: {len(direct_matches)}")
        mapping_strategies.append(('direct', dict(zip(response_ids, response_categories))))
    
    # Strategy 2: Convert response format to flow format
    # response format: "01-001" -> flow format: "PT01-1" 
    def convert_id_format(response_id):
        """Convert response ID format to flow cytometry format"""
        # Try different conversion patterns
        patterns = [
            # Pattern 1: "01-001" -> "PT01-1"
            lambda x: f"PT{x.split('-')[0]}-{x.split('-')[1].lstrip('0') or '0'}",
            # Pattern 2: "01-001" -> "01-001" (same)
            lambda x: x,
            # Pattern 3: "01-001" -> "PT01-001"
            lambda x: f"PT{x}",
            # Pattern 4: Remove leading zeros
            lambda x: f"PT{x.split('-')[0].lstrip('0') or '0'}-{x.split('-')[1].lstrip('0') or '0'}"
        ]
        
        return patterns
    
    # Test conversion patterns
    best_mapping = None
    best_match_count = 0
    
    for pattern_idx, pattern_func in enumerate(convert_id_format("")):
        try:
            converted_ids = [pattern_func(str(rid)) for rid in response_ids]
            matches = set(converted_ids) & set(flow_subjects)
            match_count = len(matches)
            
            print(f"Pattern {pattern_idx + 1}: {match_count} matches")
            if match_count > 0:
                print(f"  Sample conversions: {list(zip(response_ids.head(3), converted_ids[:3]))}")
                print(f"  Matches found: {sorted(list(matches))[:5]}")
            
            if match_count > best_match_count:
                best_match_count = match_count
                best_mapping = dict(zip(converted_ids, response_categories))
                best_pattern = pattern_idx + 1
                
        except Exception as e:
            print(f"Pattern {pattern_idx + 1} failed: {e}")
    
    if best_mapping and best_match_count > 0:
        print(f"\n✓ Best mapping found using Pattern {best_pattern}: {best_match_count} matches")
        
        # Filter to only matched subjects
        matched_mapping = {k: v for k, v in best_mapping.items() if k in flow_subjects}
        
        print(f"✓ Final mapping: {len(matched_mapping)} patients")
        
        # Show response distribution
        response_counts = pd.Series(list(matched_mapping.values())).value_counts()
        print("\nResponse distribution in matched data:")
        for resp, count in response_counts.items():
            print(f"  {resp}: {count} patients")
        
        return matched_mapping
    
    else:
        print(" No suitable mapping pattern found")
        print("Manual mapping may be required")
        
        # Show available IDs for manual inspection
        print(f"\nFlow cytometry subjects (first 10): {flow_subjects[:10].tolist()}")
        print(f"Response data IDs (first 10): {response_ids.head(10).tolist()}")
        
        return None

# ========================
# 3. RUN COMPLETE ANALYSIS
# ========================

def run_response_analysis_from_file():
    """Load response data and run complete analysis"""
    
    print("="*80)
    print("RUNNING CLINICAL RESPONSE ANALYSIS FROM FILE")
    print("="*80)
    
    # Load response data
    response_raw = load_response_data()
    
    # Create mapping
    response_mapping = create_response_mapping(response_raw)
    
    if response_mapping:
        print(f"\n✓ Successfully mapped {len(response_mapping)} patients")
        
        # Import the analysis functions from the previous artifact
        # (These would normally be imported, but for this demo we'll reference them)
        
        # Run the complete analysis using the functions from the previous artifact
        try:
            # Setup response mapping
            response_df = setup_response_mapping(response_mapping)
            
            # Run all analyses
            print("\n" + "="*60)
            print("RUNNING BASELINE ANALYSIS...")
            baseline_response, baseline_results = analyze_baseline_by_response(response_df)
            
            print("\n" + "="*60) 
            print("RUNNING LONGITUDINAL ANALYSIS...")
            flow_clean, trajectory_results = analyze_trajectories_by_response(response_df)
            
            print("\n" + "="*60)
            print("RUNNING TREATMENT RESPONSE PATTERNS...")
            response_metrics_df = analyze_treatment_response_patterns(flow_clean)
            
            print("\n" + "="*60)
            print("CREATING VISUALIZATIONS...")
            response_fig = create_response_stratified_plots(flow_clean, trajectory_results)
            plt.show()
            
            print("\n" + "="*80)
            print("CLINICAL RESPONSE ANALYSIS COMPLETE")
            print("="*80)
            
            # Return results
            analysis_results = {
                'response_mapping': response_mapping,
                'response_df': response_df,
                'baseline_response': baseline_response,
                'baseline_results': baseline_results,
                'flow_clean': flow_clean,
                'trajectory_results': trajectory_results,
                'response_metrics_df': response_metrics_df
            }
            
            return analysis_results
            
        except NameError as e:
            print(f" Analysis functions not available: {e}")
            print("Please run the clinical_response_analysis artifact first")
            return None
            
    else:
        print(" Could not create patient mapping - analysis cannot proceed")
        return None

# ========================
# 4. SUMMARY STATISTICS
# ========================

def print_analysis_summary(analysis_results):
    """Print summary of key findings"""
    
    if not analysis_results:
        return
    
    print("\n" + "="*60)
    print("KEY FINDINGS SUMMARY")
    print("="*60)
    
    response_df = analysis_results['response_df']
    baseline_results = analysis_results['baseline_results']
    
    # Overall distribution
    response_dist = response_df['Response_Category'].value_counts()
    print("Response distribution in analyzed patients:")
    for resp, count in response_dist.items():
        print(f"  {resp}: {count} patients ({count/len(response_df)*100:.1f}%)")
    
    # Baseline findings for main gate
    main_gate = 'Live+/PD-1+'
    if main_gate in baseline_results:
        print(f"\nBaseline {main_gate} levels by response:")
        gate_results = baseline_results[main_gate]
        
        for resp_cat in ['EXT', 'INT', 'NON']:
            if resp_cat in gate_results:
                data = gate_results[resp_cat]
                print(f"  {resp_cat}: {data['median']:.2f}% (median), n={data['n']}")
    
    print("\n✓ Complete analysis results available in 'analysis_results' variable")

# ========================
# RUN THE ANALYSIS
# ========================

print("Ready to analyze clinical response correlations!")
print("Run: analysis_results = run_response_analysis_from_file()")

analysis_results = run_response_analysis_from_file()


# In[44]:


# ========================
# CLINICAL RESPONSE ASSOCIATION ANALYSIS
# ========================

print("=== CLINICAL RESPONSE ASSOCIATION ANALYSIS ===")
print("Analyzing PD-1 expression patterns by treatment-free survival (TFS) categories")

# ========================
# 1. DATA SETUP AND MAPPING
# ========================

print("\n1. SETTING UP CLINICAL RESPONSE DATA...")

def setup_response_mapping(response_dict):
    """
    Set up mapping between patient IDs and response categories
    
    Parameters:
    response_dict: Dictionary mapping patient ID to response category
    Example: {'PT01-1': 'EXT', 'PT02-1': 'INT', 'PT03-1': 'NON', ...}
    """
    
    # Create response mapping dataframe
    response_df = pd.DataFrame([
        {'Subject': subject, 'Response_Category': category} 
        for subject, category in response_dict.items()
    ])
    
    print(f"Response mapping created for {len(response_df)} patients")
    
    # Validate response categories
    response_counts = response_df['Response_Category'].value_counts()
    print("\nResponse category distribution:")
    for category, count in response_counts.items():
        percentage = count / len(response_df) * 100
        print(f"  {category}: {count} patients ({percentage:.1f}%)")
    
    return response_df

# ========================
# 2. BASELINE ANALYSIS BY RESPONSE GROUP
# ========================

def analyze_baseline_by_response(response_df):
    """Analyze baseline PD-1 levels by response category"""
    
    print("\n2. BASELINE PD-1 ANALYSIS BY RESPONSE CATEGORY...")
    
    # Check if experimental_data exists
    if 'experimental_data' not in globals():
        print("❌ experimental_data not found. Please run the initial data loading section first.")
        return None, None
    
    # Merge flow data with response categories
    baseline_data = experimental_data[experimental_data['Timepoint'] == 'B1W1'].copy()
    
    # Handle duplicates
    baseline_clean = baseline_data.groupby(['Subject', 'Gate']).agg({
        'Percent_Parent': 'median',
        'Median_FI': 'median'
    }).reset_index()
    
    # Merge with response data
    baseline_response = baseline_clean.merge(response_df, on='Subject', how='inner')
    
    print(f"Baseline data available for {baseline_response['Subject'].nunique()} patients")
    
    # Analyze each gate
    baseline_results = {}
    
    for gate in baseline_response['Gate'].unique():
        gate_data = baseline_response[baseline_response['Gate'] == gate]
        
        print(f"\n{gate} - Baseline % Parent:")
        
        gate_results = {}
        for category in ['EXT', 'INT', 'NON']:
            category_data = gate_data[gate_data['Response_Category'] == category]
            if len(category_data) > 0:
                median_val = category_data['Percent_Parent'].median()
                mean_val = category_data['Percent_Parent'].mean()
                std_val = category_data['Percent_Parent'].std()
                n_val = len(category_data)
                
                gate_results[category] = {
                    'n': n_val,
                    'median': median_val,
                    'mean': mean_val,
                    'std': std_val
                }
                
                print(f"  {category}: {median_val:.2f} (median), {mean_val:.2f}±{std_val:.2f} (mean±SD), n={n_val}")
        
        # Statistical test (Kruskal-Wallis for 3 groups)
        if len(gate_results) == 3:
            try:
                from scipy.stats import kruskal
                ext_vals = gate_data[gate_data['Response_Category'] == 'EXT']['Percent_Parent']
                int_vals = gate_data[gate_data['Response_Category'] == 'INT']['Percent_Parent']
                non_vals = gate_data[gate_data['Response_Category'] == 'NON']['Percent_Parent']
                
                if len(ext_vals) > 0 and len(int_vals) > 0 and len(non_vals) > 0:
                    stat, p_val = kruskal(ext_vals, int_vals, non_vals)
                    print(f"  Kruskal-Wallis p-value: {p_val:.4f}")
                    
                    # Pairwise comparisons if significant
                    if p_val < 0.1:  # Use liberal threshold for exploration
                        from scipy.stats import mannwhitneyu
                        print("  Pairwise comparisons:")
                        
                        comparisons = [('EXT', 'NON'), ('EXT', 'INT'), ('INT', 'NON')]
                        for cat1, cat2 in comparisons:
                            vals1 = gate_data[gate_data['Response_Category'] == cat1]['Percent_Parent']
                            vals2 = gate_data[gate_data['Response_Category'] == cat2]['Percent_Parent']
                            if len(vals1) > 0 and len(vals2) > 0:
                                stat_pw, p_pw = mannwhitneyu(vals1, vals2, alternative='two-sided')
                                print(f"    {cat1} vs {cat2}: p = {p_pw:.4f}")
            
            except Exception as e:
                print(f"  Statistical test failed: {e}")
        
        baseline_results[gate] = gate_results
    
    return baseline_response, baseline_results
    """Analyze baseline PD-1 levels by response category"""
    
    print("\n2. BASELINE PD-1 ANALYSIS BY RESPONSE CATEGORY...")
    
    # Merge flow data with response categories
    baseline_data = experimental_data[experimental_data['Timepoint'] == 'B1W1'].copy()
    
    # Handle duplicates
    baseline_clean = baseline_data.groupby(['Subject', 'Gate']).agg({
        'Percent_Parent': 'median',
        'Median_FI': 'median'
    }).reset_index()
    
    # Merge with response data
    baseline_response = baseline_clean.merge(response_df, on='Subject', how='inner')
    
    print(f"Baseline data available for {baseline_response['Subject'].nunique()} patients")
    
    # Analyze each gate
    baseline_results = {}
    
    for gate in baseline_response['Gate'].unique():
        gate_data = baseline_response[baseline_response['Gate'] == gate]
        
        print(f"\n{gate} - Baseline % Parent:")
        
        gate_results = {}
        for category in ['EXT', 'INT', 'NON']:
            category_data = gate_data[gate_data['Response_Category'] == category]
            if len(category_data) > 0:
                median_val = category_data['Percent_Parent'].median()
                mean_val = category_data['Percent_Parent'].mean()
                std_val = category_data['Percent_Parent'].std()
                n_val = len(category_data)
                
                gate_results[category] = {
                    'n': n_val,
                    'median': median_val,
                    'mean': mean_val,
                    'std': std_val
                }
                
                print(f"  {category}: {median_val:.2f} (median), {mean_val:.2f}±{std_val:.2f} (mean±SD), n={n_val}")
        
        # Statistical test (Kruskal-Wallis for 3 groups)
        if len(gate_results) == 3:
            try:
                from scipy.stats import kruskal
                ext_vals = gate_data[gate_data['Response_Category'] == 'EXT']['Percent_Parent']
                int_vals = gate_data[gate_data['Response_Category'] == 'INT']['Percent_Parent']
                non_vals = gate_data[gate_data['Response_Category'] == 'NON']['Percent_Parent']
                
                if len(ext_vals) > 0 and len(int_vals) > 0 and len(non_vals) > 0:
                    stat, p_val = kruskal(ext_vals, int_vals, non_vals)
                    print(f"  Kruskal-Wallis p-value: {p_val:.4f}")
                    
                    # Pairwise comparisons if significant
                    if p_val < 0.1:  # Use liberal threshold for exploration
                        from scipy.stats import mannwhitneyu
                        print("  Pairwise comparisons:")
                        
                        comparisons = [('EXT', 'NON'), ('EXT', 'INT'), ('INT', 'NON')]
                        for cat1, cat2 in comparisons:
                            vals1 = gate_data[gate_data['Response_Category'] == cat1]['Percent_Parent']
                            vals2 = gate_data[gate_data['Response_Category'] == cat2]['Percent_Parent']
                            if len(vals1) > 0 and len(vals2) > 0:
                                stat_pw, p_pw = mannwhitneyu(vals1, vals2, alternative='two-sided')
                                print(f"    {cat1} vs {cat2}: p = {p_pw:.4f}")
            
            except Exception as e:
                print(f"  Statistical test failed: {e}")
        
        baseline_results[gate] = gate_results
    
    return baseline_response, baseline_results

# ========================
# 3. LONGITUDINAL ANALYSIS BY RESPONSE GROUP
# ========================

def analyze_trajectories_by_response(response_df):
    """Analyze PD-1 trajectories by response category"""
    
    print("\n3. LONGITUDINAL TRAJECTORY ANALYSIS BY RESPONSE...")
    
    # Check if experimental_data exists
    if 'experimental_data' not in globals():
        print("❌ experimental_data not found. Please run the initial data loading section first.")
        return None, None
    
    # Merge all timepoint data with response categories
    flow_response = experimental_data.merge(response_df, on='Subject', how='inner')
    
    # Handle duplicates
    flow_clean = flow_response.groupby(['Subject', 'Timepoint', 'Gate']).agg({
        'Percent_Parent': 'median',
        'Median_FI': 'median',
        'Response_Category': 'first'
    }).reset_index()
    
    print(f"Longitudinal data available for {flow_clean['Subject'].nunique()} patients")
    
    # Add timepoint numbers
    timepoint_map = {'B1W1': 0, 'B2W2': 1, 'B2W4': 2, 'B4W1': 3}
    flow_clean['Timepoint_num'] = flow_clean['Timepoint'].map(timepoint_map)
    
    # Analyze each gate
    trajectory_results = {}
    
    for gate in flow_clean['Gate'].unique():
        gate_data = flow_clean[flow_clean['Gate'] == gate]
        
        print(f"\n{gate} - Trajectory Analysis:")
        
        gate_traj = {}
        for category in ['EXT', 'INT', 'NON']:
            category_data = gate_data[gate_data['Response_Category'] == category]
            
            if len(category_data) > 0:
                # Calculate summary by timepoint
                traj_summary = category_data.groupby('Timepoint')['Percent_Parent'].agg([
                    'count', 'median', 'mean', 'std'
                ]).round(3)
                
                gate_traj[category] = traj_summary
                
                print(f"\n  {category} Response Group:")
                for tp in ['B1W1', 'B2W2', 'B2W4', 'B4W1']:
                    if tp in traj_summary.index:
                        row = traj_summary.loc[tp]
                        print(f"    {tp}: {row['median']:.2f} (median), n={int(row['count'])}")
                
                # Calculate fold changes from baseline
                if 'B1W1' in traj_summary.index:
                    baseline = traj_summary.loc['B1W1', 'median']
                    print(f"    Fold changes from baseline:")
                    for tp in ['B2W2', 'B2W4', 'B4W1']:
                        if tp in traj_summary.index:
                            current = traj_summary.loc[tp, 'median']
                            fold_change = current / baseline if baseline > 0 else float('inf')
                            percent_change = ((current - baseline) / baseline) * 100 if baseline > 0 else float('inf')
                            print(f"      {tp}: {fold_change:.2f}x ({percent_change:+.1f}%)")
        
        trajectory_results[gate] = gate_traj
    
    return flow_clean, trajectory_results
    """Analyze PD-1 trajectories by response category"""
    
    print("\n3. LONGITUDINAL TRAJECTORY ANALYSIS BY RESPONSE...")
    
    # Merge all timepoint data with response categories
    flow_response = experimental_data.merge(response_df, on='Subject', how='inner')
    
    # Handle duplicates
    flow_clean = flow_response.groupby(['Subject', 'Timepoint', 'Gate']).agg({
        'Percent_Parent': 'median',
        'Median_FI': 'median',
        'Response_Category': 'first'
    }).reset_index()
    
    print(f"Longitudinal data available for {flow_clean['Subject'].nunique()} patients")
    
    # Add timepoint numbers
    timepoint_map = {'B1W1': 0, 'B2W2': 1, 'B2W4': 2, 'B4W1': 3}
    flow_clean['Timepoint_num'] = flow_clean['Timepoint'].map(timepoint_map)
    
    # Analyze each gate
    trajectory_results = {}
    
    for gate in flow_clean['Gate'].unique():
        gate_data = flow_clean[flow_clean['Gate'] == gate]
        
        print(f"\n{gate} - Trajectory Analysis:")
        
        gate_traj = {}
        for category in ['EXT', 'INT', 'NON']:
            category_data = gate_data[gate_data['Response_Category'] == category]
            
            if len(category_data) > 0:
                # Calculate summary by timepoint
                traj_summary = category_data.groupby('Timepoint')['Percent_Parent'].agg([
                    'count', 'median', 'mean', 'std'
                ]).round(3)
                
                gate_traj[category] = traj_summary
                
                print(f"\n  {category} Response Group:")
                for tp in ['B1W1', 'B2W2', 'B2W4', 'B4W1']:
                    if tp in traj_summary.index:
                        row = traj_summary.loc[tp]
                        print(f"    {tp}: {row['median']:.2f} (median), n={int(row['count'])}")
                
                # Calculate fold changes from baseline
                if 'B1W1' in traj_summary.index:
                    baseline = traj_summary.loc['B1W1', 'median']
                    print(f"    Fold changes from baseline:")
                    for tp in ['B2W2', 'B2W4', 'B4W1']:
                        if tp in traj_summary.index:
                            current = traj_summary.loc[tp, 'median']
                            fold_change = current / baseline if baseline > 0 else float('inf')
                            percent_change = ((current - baseline) / baseline) * 100 if baseline > 0 else float('inf')
                            print(f"      {tp}: {fold_change:.2f}x ({percent_change:+.1f}%)")
        
        trajectory_results[gate] = gate_traj
    
    return flow_clean, trajectory_results

# ========================
# 4. TREATMENT RESPONSE PATTERN ANALYSIS
# ========================

def analyze_treatment_response_patterns(flow_clean):
    """Analyze specific treatment response patterns"""
    
    print("\n4. TREATMENT RESPONSE PATTERN ANALYSIS...")
    
    # Focus on main gate
    main_gate = 'Live+/PD-1+'
    if main_gate not in flow_clean['Gate'].unique():
        main_gate = flow_clean['Gate'].unique()[0]
    
    main_data = flow_clean[flow_clean['Gate'] == main_gate].copy()
    
    print(f"Analyzing {main_gate} response patterns:")
    
    # Calculate individual patient response metrics
    patient_responses = []
    
    for subject in main_data['Subject'].unique():
        subject_data = main_data[main_data['Subject'] == subject]
        response_cat = subject_data['Response_Category'].iloc[0]
        
        # Get values at each timepoint
        tp_values = {}
        for tp in ['B1W1', 'B2W2', 'B2W4', 'B4W1']:
            tp_data = subject_data[subject_data['Timepoint'] == tp]
            if len(tp_data) > 0:
                tp_values[tp] = tp_data['Percent_Parent'].iloc[0]
        
        # Calculate response metrics if baseline available
        if 'B1W1' in tp_values:
            baseline = tp_values['B1W1']
            
            metrics = {
                'Subject': subject,
                'Response_Category': response_cat,
                'Baseline': baseline
            }
            
            # Calculate changes
            for tp in ['B2W2', 'B2W4', 'B4W1']:
                if tp in tp_values:
                    current = tp_values[tp]
                    metrics[f'{tp}_Value'] = current
                    metrics[f'{tp}_FoldChange'] = current / baseline if baseline > 0 else None
                    metrics[f'{tp}_PercentChange'] = ((current - baseline) / baseline) * 100 if baseline > 0 else None
            
            patient_responses.append(metrics)
    
    response_metrics_df = pd.DataFrame(patient_responses)
    
    # Analyze response patterns
    print(f"\nIndividual Patient Response Patterns ({main_gate}):")
    
    for category in ['EXT', 'INT', 'NON']:
        cat_data = response_metrics_df[response_metrics_df['Response_Category'] == category]
        if len(cat_data) > 0:
            print(f"\n{category} Responders (n={len(cat_data)}):")
            
            # Baseline levels
            baseline_median = cat_data['Baseline'].median()
            print(f"  Baseline median: {baseline_median:.2f}%")
            
            # Response patterns
            for tp in ['B2W2', 'B2W4', 'B4W1']:
                fc_col = f'{tp}_FoldChange'
                pc_col = f'{tp}_PercentChange'
                
                if fc_col in cat_data.columns:
                    fc_vals = cat_data[fc_col].dropna()
                    pc_vals = cat_data[pc_col].dropna()
                    
                    if len(fc_vals) > 0:
                        fc_median = fc_vals.median()
                        pc_median = pc_vals.median()
                        n_available = len(fc_vals)
                        print(f"  {tp}: {fc_median:.2f}x ({pc_median:+.1f}%), n={n_available}")
    
    return response_metrics_df
    """Analyze specific treatment response patterns"""
    
    print("\n4. TREATMENT RESPONSE PATTERN ANALYSIS...")
    
    # Focus on main gate
    main_gate = 'Live+/PD-1+'
    if main_gate not in flow_clean['Gate'].unique():
        main_gate = flow_clean['Gate'].unique()[0]
    
    main_data = flow_clean[flow_clean['Gate'] == main_gate].copy()
    
    print(f"Analyzing {main_gate} response patterns:")
    
    # Calculate individual patient response metrics
    patient_responses = []
    
    for subject in main_data['Subject'].unique():
        subject_data = main_data[main_data['Subject'] == subject]
        response_cat = subject_data['Response_Category'].iloc[0]
        
        # Get values at each timepoint
        tp_values = {}
        for tp in ['B1W1', 'B2W2', 'B2W4', 'B4W1']:
            tp_data = subject_data[subject_data['Timepoint'] == tp]
            if len(tp_data) > 0:
                tp_values[tp] = tp_data['Percent_Parent'].iloc[0]
        
        # Calculate response metrics if baseline available
        if 'B1W1' in tp_values:
            baseline = tp_values['B1W1']
            
            metrics = {
                'Subject': subject,
                'Response_Category': response_cat,
                'Baseline': baseline
            }
            
            # Calculate changes
            for tp in ['B2W2', 'B2W4', 'B4W1']:
                if tp in tp_values:
                    current = tp_values[tp]
                    metrics[f'{tp}_Value'] = current
                    metrics[f'{tp}_FoldChange'] = current / baseline if baseline > 0 else None
                    metrics[f'{tp}_PercentChange'] = ((current - baseline) / baseline) * 100 if baseline > 0 else None
            
            patient_responses.append(metrics)
    
    response_metrics_df = pd.DataFrame(patient_responses)
    
    # Analyze response patterns
    print(f"\nIndividual Patient Response Patterns ({main_gate}):")
    
    for category in ['EXT', 'INT', 'NON']:
        cat_data = response_metrics_df[response_metrics_df['Response_Category'] == category]
        if len(cat_data) > 0:
            print(f"\n{category} Responders (n={len(cat_data)}):")
            
            # Baseline levels
            baseline_median = cat_data['Baseline'].median()
            print(f"  Baseline median: {baseline_median:.2f}%")
            
            # Response patterns
            for tp in ['B2W2', 'B2W4', 'B4W1']:
                fc_col = f'{tp}_FoldChange'
                pc_col = f'{tp}_PercentChange'
                
                if fc_col in cat_data.columns:
                    fc_vals = cat_data[fc_col].dropna()
                    pc_vals = cat_data[pc_col].dropna()
                    
                    if len(fc_vals) > 0:
                        fc_median = fc_vals.median()
                        pc_median = pc_vals.median()
                        n_available = len(fc_vals)
                        print(f"  {tp}: {fc_median:.2f}x ({pc_median:+.1f}%), n={n_available}")
    
    return response_metrics_df

# ========================
# 5. VISUALIZATION BY RESPONSE GROUP
# ========================

def create_response_stratified_plots(flow_clean, trajectory_results):
    """Create visualizations stratified by response category"""
    
    print("\n5. CREATING RESPONSE-STRATIFIED VISUALIZATIONS...")
    
    gates = flow_clean['Gate'].unique()
    n_gates = len(gates)
    
    # Create figure with subplots for each gate
    fig, axes = plt.subplots(n_gates, 2, figsize=(16, 4*n_gates))
    if n_gates == 1:
        axes = axes.reshape(1, -1)
    
    # Color scheme for response categories
    response_colors = {'EXT': 'green', 'INT': 'blue', 'NON': 'red'}
    response_labels = {
        'EXT': 'EXT',
        'INT': 'INT', 
        'NON': 'NON'
    }
    
    for i, gate in enumerate(gates):
        gate_data = flow_clean[flow_clean['Gate'] == gate]
        
        # Plot 1: % Parent trajectories by response group
        ax1 = axes[i, 0]
        
        for category in ['EXT', 'INT', 'NON']:
            cat_data = gate_data[gate_data['Response_Category'] == category]
            
            if len(cat_data) > 0:
                # Individual trajectories (light lines)
                subjects = cat_data['Subject'].unique()
                for subject in subjects:
                    subj_data = cat_data[cat_data['Subject'] == subject].sort_values('Timepoint_num')
                    if len(subj_data) >= 2:
                        ax1.plot(subj_data['Timepoint_num'], subj_data['Percent_Parent'],
                                'o-', alpha=0.2, color=response_colors[category], linewidth=1, markersize=3)
                
                # Mean trajectory (bold line)
                mean_data = cat_data.groupby('Timepoint_num')['Percent_Parent'].agg(['mean', 'sem', 'count']).reset_index()
                ax1.errorbar(mean_data['Timepoint_num'], mean_data['mean'], yerr=mean_data['sem'],
                            fmt='o-', color=response_colors[category], linewidth=3, markersize=8, 
                            capsize=5, capthick=2, label=response_labels[category])
        
        ax1.set_title(f'{gate} - % Parent by Response Category')
        ax1.set_xlabel('Treatment Phase')
        ax1.set_ylabel('% of Parent')
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(['B1W1', 'B2W2', 
                            'B2W4', 'B4W1'])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Median FI trajectories by response group
        ax2 = axes[i, 1]
        
        for category in ['EXT', 'INT', 'NON']:
            cat_data = gate_data[gate_data['Response_Category'] == category]
            
            if len(cat_data) > 0:
                # Individual trajectories
                subjects = cat_data['Subject'].unique()
                for subject in subjects:
                    subj_data = cat_data[cat_data['Subject'] == subject].sort_values('Timepoint_num')
                    if len(subj_data) >= 2:
                        ax2.plot(subj_data['Timepoint_num'], subj_data['Median_FI'],
                                'o-', alpha=0.2, color=response_colors[category], linewidth=1, markersize=3)
                
                # Mean trajectory
                mean_data = cat_data.groupby('Timepoint_num')['Median_FI'].agg(['mean', 'sem']).reset_index()
                ax2.errorbar(mean_data['Timepoint_num'], mean_data['mean'], yerr=mean_data['sem'],
                            fmt='o-', color=response_colors[category], linewidth=3, markersize=8,
                            capsize=5, capthick=2, label=response_labels[category])
        
        ax2.set_title(f'{gate} - Median FI by Response Category')
        ax2.set_xlabel('Treatment Phase')
        ax2.set_ylabel('Median FI')
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['B1W1', 'B2W2', 
                            'B2W4', 'B4W1'])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    return fig
    """Create visualizations stratified by response category"""
    
    print("\n5. CREATING RESPONSE-STRATIFIED VISUALIZATIONS...")
    
    gates = flow_clean['Gate'].unique()
    n_gates = len(gates)
    
    # Create figure with subplots for each gate
    fig, axes = plt.subplots(n_gates, 2, figsize=(16, 4*n_gates))
    if n_gates == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('PD-1 Expression Trajectories by Clinical Response Category\n' +
                 'EXT: TFS >5yr | INT: TFS 6mo-5yr | NON: TFS <6mo', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme for response categories
    response_colors = {'EXT': 'green', 'INT': 'orange', 'NON': 'red'}
    response_labels = {
        'EXT': 'Extreme Responders (>5yr TFS)',
        'INT': 'Intermediate Responders (6mo-5yr TFS)', 
        'NON': 'Non-Responders (<6mo TFS)'
    }
    
    for i, gate in enumerate(gates):
        gate_data = flow_clean[flow_clean['Gate'] == gate]
        
        # Plot 1: % Parent trajectories by response group
        ax1 = axes[i, 0]
        
        for category in ['EXT', 'INT', 'NON']:
            cat_data = gate_data[gate_data['Response_Category'] == category]
            
            if len(cat_data) > 0:
                # Individual trajectories (light lines)
                subjects = cat_data['Subject'].unique()
                for subject in subjects:
                    subj_data = cat_data[cat_data['Subject'] == subject].sort_values('Timepoint_num')
                    if len(subj_data) >= 2:
                        ax1.plot(subj_data['Timepoint_num'], subj_data['Percent_Parent'],
                                'o-', alpha=0.2, color=response_colors[category], linewidth=1, markersize=3)
                
                # Mean trajectory (bold line)
                mean_data = cat_data.groupby('Timepoint_num')['Percent_Parent'].agg(['mean', 'sem', 'count']).reset_index()
                ax1.errorbar(mean_data['Timepoint_num'], mean_data['mean'], yerr=mean_data['sem'],
                            fmt='o-', color=response_colors[category], linewidth=3, markersize=8, 
                            capsize=5, capthick=2, label=response_labels[category])
        
        ax1.set_title(f'{gate} - % Parent by Response Category')
        ax1.set_xlabel('Treatment Phase')
        ax1.set_ylabel('% of Parent')
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(['B1W1', 'B2W2', 
                            'B2W4', 'B4W1'])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Median FI trajectories by response group
        ax2 = axes[i, 1]
        
        for category in ['EXT', 'INT', 'NON']:
            cat_data = gate_data[gate_data['Response_Category'] == category]
            
            if len(cat_data) > 0:
                # Individual trajectories
                subjects = cat_data['Subject'].unique()
                for subject in subjects:
                    subj_data = cat_data[cat_data['Subject'] == subject].sort_values('Timepoint_num')
                    if len(subj_data) >= 2:
                        ax2.plot(subj_data['Timepoint_num'], subj_data['Median_FI'],
                                'o-', alpha=0.2, color=response_colors[category], linewidth=1, markersize=3)
                
                # Mean trajectory
                mean_data = cat_data.groupby('Timepoint_num')['Median_FI'].agg(['mean', 'sem']).reset_index()
                ax2.errorbar(mean_data['Timepoint_num'], mean_data['mean'], yerr=mean_data['sem'],
                            fmt='o-', color=response_colors[category], linewidth=3, markersize=8,
                            capsize=5, capthick=2, label=response_labels[category])
        
        ax2.set_title(f'{gate} - Median FI by Response Category')
        ax2.set_xlabel('Treatment Phase')
        ax2.set_ylabel('Median FI')
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['B1W1', 'B2W2', 
                            'B2W4', 'B4W1'])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    # Save the response-stratified trajectory figure at 600 DPI
    return fig
    fig = plt.gcf()
    filename = 'PD1_trajectories_test.png'
    fig.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')


# ========================
# 6. SUMMARY ANALYSIS FUNCTION
# ========================

def run_complete_response_analysis(response_data):
    """Run complete analysis with response data"""
    
    print("="*80)
    print("RUNNING COMPLETE CLINICAL RESPONSE ANALYSIS")
    print("="*80)
    
    # Setup response mapping
    response_df = setup_response_mapping(response_data)
    
    # Run all analyses
    baseline_response, baseline_results = analyze_baseline_by_response(response_df)
    flow_clean, trajectory_results = analyze_trajectories_by_response(response_df)
    response_metrics_df = analyze_treatment_response_patterns(flow_clean)
    
    # Create visualizations
    response_fig = create_response_stratified_plots(flow_clean, trajectory_results)
    plt.show()

    
    print("\n" + "="*80)
    print("CLINICAL RESPONSE ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'response_df': response_df,
        'baseline_response': baseline_response,
        'baseline_results': baseline_results,
        'flow_clean': flow_clean,
        'trajectory_results': trajectory_results,
        'response_metrics_df': response_metrics_df
    }
    """Analyze baseline PD-1 as predictive biomarker"""
    
    print("\n6. PREDICTIVE BIOMARKER ANALYSIS...")
    
    # Analyze baseline PD-1 as predictor of response
    main_gate = 'Live+/PD-1+'
    if main_gate not in baseline_response['Gate'].unique():
        main_gate = baseline_response['Gate'].unique()[0]
    
    baseline_main = baseline_response[baseline_response['Gate'] == main_gate]
    
    print(f"Baseline {main_gate} as Predictive Biomarker:")
    
    # ROC analysis for extreme vs non-responders
    try:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        
        # Create binary outcome: EXT vs others
        baseline_main['EXT_Response'] = (baseline_main['Response_Category'] == 'EXT').astype(int)
        
        # ROC for baseline PD-1
        if len(baseline_main) > 5:  # Need sufficient data
            fpr, tpr, thresholds = roc_curve(baseline_main['EXT_Response'], baseline_main['Percent_Parent'])
            roc_auc = auc(fpr, tpr)
            
            print(f"  Baseline % Parent ROC AUC for extreme response: {roc_auc:.3f}")
            
            # Find optimal threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            sensitivity = tpr[optimal_idx]
            specificity = 1 - fpr[optimal_idx]
            
            print(f"  Optimal threshold: {optimal_threshold:.2f}%")
            print(f"  Sensitivity: {sensitivity:.3f}")
            print(f"  Specificity: {specificity:.3f}")
            
            # Create ROC plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: Baseline {main_gate} % Parent\nPredicting Extreme Response (TFS >5yr)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.show()
    
    except ImportError:
        print("  ROC analysis requires sklearn - install with 'pip install scikit-learn'")
    
    # Threshold analysis
    print(f"\n  Threshold Analysis:")
    percentiles = [25, 50, 75]
    for p in percentiles:
        threshold = baseline_main['Percent_Parent'].quantile(p/100)
        
        high_baseline = baseline_main[baseline_main['Percent_Parent'] >= threshold]
        ext_rate_high = (high_baseline['Response_Category'] == 'EXT').mean() * 100
        n_high = len(high_baseline)
        
        low_baseline = baseline_main[baseline_main['Percent_Parent'] < threshold]
        ext_rate_low = (low_baseline['Response_Category'] == 'EXT').mean() * 100
        n_low = len(low_baseline)
        
        print(f"    {p}th percentile ({threshold:.2f}%):")
        print(f"      High baseline (≥{threshold:.2f}%): {ext_rate_high:.1f}% extreme response (n={n_high})")
        print(f"      Low baseline (<{threshold:.2f}%): {ext_rate_low:.1f}% extreme response (n={n_low})")

# ========================
# 7. SUMMARY ANALYSIS FUNCTION
# ========================

def run_complete_response_analysis(response_data):
    """Run complete analysis with response data"""
    
    print("="*80)
    print("RUNNING COMPLETE CLINICAL RESPONSE ANALYSIS")
    print("="*80)
    
    # Setup response mapping
    response_df = setup_response_mapping(response_data)
    
    # Run all analyses
    baseline_response, baseline_results = analyze_baseline_by_response(response_df)
    flow_clean, trajectory_results = analyze_trajectories_by_response(response_df)
    response_metrics_df = analyze_treatment_response_patterns(flow_clean)
    
    # Create visualizations
    response_fig = create_response_stratified_plots(flow_clean, trajectory_results)
    plt.show()
    
    print("\n" + "="*80)
    print("CLINICAL RESPONSE ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'response_df': response_df,
        'baseline_response': baseline_response,
        'baseline_results': baseline_results,
        'flow_clean': flow_clean,
        'trajectory_results': trajectory_results,
        'response_metrics_df': response_metrics_df
    }

print("\n" + "="*60)
print("CLINICAL RESPONSE ANALYSIS FUNCTIONS READY")
print("="*60)

analysis_results = run_response_analysis_from_file()


# In[46]:


# =============================================================================
# TIME-DEPENDENT HEATMAP FOR PD-1 FLOW CYTOMETRY DATA
# Complete standalone analysis
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=== TIME-DEPENDENT HEATMAP ANALYSIS ===")
print("PD-1 Expression Dynamics Across Treatment Timeline")

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

print("\n1. LOADING DATA...")

# Load flow cytometry data
try:
   flow_file = "18536 PD-1 flow data pivot 250611.xlsx"
   data_sheet = pd.read_excel(flow_file, sheet_name='Data')
   lookup_sheet = pd.read_excel(flow_file, sheet_name='SmplLookup')
   print(f"✓ Flow data loaded: {data_sheet.shape[0]} rows")
except FileNotFoundError:
   print("❌ Flow cytometry file not found")
   exit()

# Load response data
try:
   response_file = "ID response correlation.xlsx"
   response_raw = pd.read_excel(response_file)
   print(f"✓ Response data loaded: {len(response_raw)} patients")
except FileNotFoundError:
   print("❌ Response file not found")
   exit()

# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================

print("\n2. PREPROCESSING DATA...")

# Clean flow cytometry data
data_sheet.columns = [col.strip() for col in data_sheet.columns]
data_sheet['Percent_Parent'] = pd.to_numeric(data_sheet['% of parent'], errors='coerce')
data_sheet['Median_FI'] = pd.to_numeric(data_sheet['Median'], errors='coerce')
data_sheet['Gate'] = data_sheet['Gate name']

# Clean lookup data
lookup_sheet.columns = [col.strip() for col in lookup_sheet.columns]

# Merge flow data with lookup
merged_data = data_sheet.merge(lookup_sheet, left_on='Filename', right_on='Flow ID', how='left')

# Separate experimental from controls
experimental_data = merged_data[~merged_data['Subject'].isin(['FMO Ctrl', 'PBMC Ctrl'])].copy()

print(f"✓ Experimental data: {len(experimental_data)} rows")

# =============================================================================
# 3. CREATE RESPONSE MAPPING
# =============================================================================

print("\n3. MAPPING PATIENT RESPONSES...")

# Create response mapping
response_raw.columns = [col.strip() for col in response_raw.columns]
response_ids = response_raw.iloc[:, 0].astype(str)
response_categories = response_raw.iloc[:, 1]

# Map response IDs to flow cytometry subject IDs (direct match for this dataset)
response_mapping = dict(zip(response_ids, response_categories))

# Filter to subjects present in flow data
flow_subjects = experimental_data['Subject'].unique()
flow_subjects = [str(s) for s in flow_subjects if pd.notna(s)]

matched_mapping = {k: v for k, v in response_mapping.items() if k in flow_subjects}
print(f"✓ Matched {len(matched_mapping)} patients to response categories")

# Create response dataframe
response_df = pd.DataFrame([
   {'Subject': subject, 'Response_Category': category} 
   for subject, category in matched_mapping.items()
])

# =============================================================================
# 4. PREPARE FLOW DATA FOR ANALYSIS
# =============================================================================

print("\n4. PREPARING FLOW DATA...")

# Merge experimental data with response categories
flow_response = experimental_data.merge(response_df, on='Subject', how='inner')

# Handle duplicates by taking median
flow_clean = flow_response.groupby(['Subject', 'Timepoint', 'Gate']).agg({
   'Percent_Parent': 'median',
   'Median_FI': 'median',
   'Response_Category': 'first'
}).reset_index()

# Add timepoint numbers for ordering
timepoint_map = {'B1W1': 0, 'B2W2': 1, 'B2W4': 2, 'B4W1': 3}
flow_clean['Timepoint_num'] = flow_clean['Timepoint'].map(timepoint_map)

print(f"✓ Clean flow data: {len(flow_clean)} data points")
print(f"✓ Subjects: {flow_clean['Subject'].nunique()}")
print(f"✓ Gates: {list(flow_clean['Gate'].unique())}")

# Display response distribution
response_dist = flow_clean['Response_Category'].value_counts()
print(f"\nResponse distribution:")
for resp, count in response_dist.items():
   n_subjects = flow_clean[flow_clean['Response_Category'] == resp]['Subject'].nunique()
   print(f"  {resp}: {n_subjects} subjects")

# =============================================================================
# 5. CREATE TIME-DEPENDENT HEATMAP
# =============================================================================

def create_combined_time_dependent_heatmap(flow_clean):
    """Create single heatmap showing all response groups together"""
    
    # Create pivot table for heatmap data
    heatmap_data = flow_clean.groupby(['Response_Category', 'Gate', 'Timepoint'])['Percent_Parent'].median().reset_index()
    
    # Create combined labels for rows (Gate_ResponseGroup)
    heatmap_data['Gate_Response'] = heatmap_data['Gate'] + '_' + heatmap_data['Response_Category']
    
    # Create the main pivot table
    timepoints = ['B1W1', 'B2W2', 'B2W4', 'B4W1']
    pivot_data = heatmap_data.pivot(index='Gate_Response', columns='Timepoint', values='Percent_Parent')
    pivot_data = pivot_data.reindex(columns=timepoints)
    
    # Sort rows to group by gate, then by response category
    gates = sorted(flow_clean['Gate'].unique())
    response_groups = ['EXT', 'INT', 'NON']
    
    # Create ordered row index
    ordered_rows = []
    for gate in gates:
        for resp in response_groups:
            row_name = f"{gate}_{resp}"
            if row_name in pivot_data.index:
                ordered_rows.append(row_name)
    
    pivot_data = pivot_data.reindex(ordered_rows)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, max(8, len(ordered_rows) * 0.4)))
    
    # Create heatmap with your specified parameters
    sns.heatmap(pivot_data, 
               annot=False,  # Remove numbers
               cmap='Blues_r',  # Your color scheme
               linewidths=0,  # Remove lines
               ax=ax,
               cbar_kws={'label': '% of Parent Population'})
    
    # Customize appearance
    ax.set_xlabel('Treatment Timepoint', fontsize=12)
    ax.set_ylabel('Flow Gates by Response Category', fontsize=12)
    
    # Clean up row labels for better readability
    new_labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        if '_' in text:
            gate, resp = text.split('_')
            # Create cleaner labels
            resp_labels = {'EXT': 'EXT', 'INT': 'INT', 'NON': 'NON'}
            new_label = f"{gate} ({resp_labels.get(resp, resp)})"
            new_labels.append(new_label)
        else:
            new_labels.append(text)
    
    ax.set_yticklabels(new_labels, rotation=0)
    
    plt.tight_layout()
    
    return fig, pivot_data

# Create the combined heatmap
print("\n5. CREATING COMBINED TIME-DEPENDENT HEATMAP...")

combined_heatmap_fig, combined_data = create_combined_time_dependent_heatmap(flow_clean)
plt.show()

# Save the figure
combined_heatmap_fig.savefig('PD1_combined_time_heatmap_clean.png', dpi=600, bbox_inches='tight', facecolor='white')
print("✓ Combined time-dependent heatmap saved as 'PD1_combined_time_heatmap_clean.png'")

print("\n" + "="*60)
print("TIME-DEPENDENT HEATMAP ANALYSIS COMPLETE")
print("="*60)


# In[41]:


def create_baseline_heatmap_by_patient(flow_clean):
    """Create baseline heatmap showing individual patients grouped by response category"""
    
    # Filter to baseline data only (B1W1)
    baseline_data = flow_clean[flow_clean['Timepoint'] == 'B1W1'].copy()
    
    # Create pivot table: Gates as rows, Individual patients as columns
    pivot_data = baseline_data.pivot(index='Gate', columns='Subject', values='Percent_Parent')
    
    # Get response mapping for each patient
    patient_response = baseline_data[['Subject', 'Response_Category']].drop_duplicates()
    patient_response_dict = dict(zip(patient_response['Subject'], patient_response['Response_Category']))
    
    # Sort patients by response category, then by patient ID
    response_order = ['EXT', 'INT', 'NON']
    ordered_patients = []
    
    for response in response_order:
        response_patients = [p for p in pivot_data.columns if patient_response_dict.get(p) == response]
        # Sort patient IDs within each response group
        response_patients = sorted(response_patients)
        ordered_patients.extend(response_patients)
    
    # Custom gate order - swap FoxP3+Treg/PD-1+ and Live+/PD-1+
    original_gates = sorted(pivot_data.index)
    
    # Reorder columns by response group
    pivot_data = pivot_data.reindex(columns=ordered_patients)
    
    # Define custom order with FoxP3 and Live swapped
    gate_order = []
    for gate in original_gates:
        if gate == 'Live+/PD-1+':
            gate_order.append('FoxP3+Treg/PD-1+')
        elif gate == 'FoxP3+Treg/PD-1+':
            gate_order.append('Live+/PD-1+')
        else:
            gate_order.append(gate)
    
    # Reorder rows
    pivot_data = pivot_data.reindex(gate_order)
    
    # Convert to numpy array and handle NaN values
    expression_matrix = pivot_data.values
    expression_matrix = np.nan_to_num(expression_matrix, nan=0.0)
    
    # Create figure using imshow
    plt.figure(figsize=(max(15, len(ordered_patients) * 0.6), 8))
    im = plt.imshow(expression_matrix, aspect='auto', cmap='Blues_r', interpolation='none')
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Baseline PD-1 Expression (% of Parent)', fontsize=12)
    
    # Set up row labels (gates)
    plt.yticks(range(len(gate_order)), gate_order, fontsize=11)
    
    # Set up column labels (patients with response category) - CENTERED
    patient_labels = []
    for patient in ordered_patients:
        response = patient_response_dict.get(patient, 'UNK')
        label = f"{patient}\n({response})"
        patient_labels.append(label)
    
    # Key fix: Use ha='center' for proper centering
    plt.xticks(range(len(ordered_patients)), patient_labels, fontsize=8, rotation=45, ha='center')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf(), pivot_data, patient_response_dict

# Replace the existing function call:
print("\n5. CREATING BASELINE HEATMAP BY PATIENT...")

baseline_patient_fig, baseline_patient_data, patient_responses = create_baseline_heatmap_by_patient(flow_clean)
plt.show()

# Save the figure
baseline_patient_fig.savefig('PD1_baseline_heatmap_by_patient_clean.png', dpi=600, bbox_inches='tight')
baseline_patient_fig.savefig('PD1_baseline_heatmap_by_patient_clean.pdf', bbox_inches='tight')
print("✓ Baseline patient heatmap saved")


# In[ ]:




