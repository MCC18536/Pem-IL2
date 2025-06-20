# Pembrolizumab + IL2 Clinical Trial Data Analysis (NCT02964078)

A comprehensive analysis pipeline for multi-omics biomarker discovery in patients treated with sequential pembrolizumab and high-dose IL-2 for metastatic renal cell carcinoma.

## Study Overview

This repository contains analysis code for a clinical trial investigating sequential pembrolizumab followed by high-dose IL-2 therapy in patients with metastatic renal cell carcinoma. The study evaluates treatment-free survival (TFS) outcomes and identifies biomarkers associated with response.

**Clinical Trial:** NCT02964078  
**Treatment Schedule:**
- **B1W1**: Baseline
- **B2W2**: Post-4 doses pembrolizumab monotherapy
- **B2W4**: Post-10 doses IL-2
- **B4W1**: After completion of IL-2

**Response Categories:**
- **EXT**: Extreme responders (TFS >5 years)
- **INT**: Intermediate responders (TFS 6 months - 5 years)
- **NON**: Non-responders (TFS <6 months)

## Repository Structure

### Core Analysis Modules

#### 1. Flow Cytometry Analysis (`Flow Cytometry PemIL2.py`)
- **Purpose**: Comprehensive flow cytometry analysis with statistical testing
- **Features**:
  - Quality control and data validation
  - Differential expression analysis between response groups
  - Multiple testing correction (FDR)
  - Volcano plots and heatmaps
  - Statistical assumption checking
- **Key Functions**:
  - `FlowAnalyzer`: Main analysis class
  - `run_flow_analysis()`: Complete pipeline execution
  - `create_results_summary()`: Generate comprehensive results

#### 2. Gene Set Enrichment Analysis (`GSEA PemIL2.py`)
- **Purpose**: Pathway enrichment analysis using GSEA
- **Features**:
  - Corrected log2 fold change calculations
  - Multiple gene set databases (GO, KEGG, Reactome, WikiPathway)
  - Enhanced visualization with publication-quality plots
  - Tiered results by FDR significance levels
- **Key Functions**:
  - `run_gsea_analysis_corrected()`: Main GSEA pipeline
  - `create_classic_pathway_plots()`: Visualization generation
  - `save_tiered_gsea_results()`: Results stratification

#### 3. NanoString Analysis (`NanoString PemIL2.py`)
- **Purpose**: Comprehensive NanoString gene expression analysis
- **Features**:
  - Data verification and quality control
  - Corrected fold change calculations for log2-transformed data
  - Multi-group statistical comparisons
  - Enhanced volcano plots and box plots
- **Key Components**:
  - **Phase 1**: Data verification and diagnostic analysis
  - **Phase 2**: Differential expression analysis
  - **Phase 3**: Visualization and result interpretation

#### 4. Olink Proteomics Analysis (`Olink PemIL2.py`)
- **Purpose**: Olink proteomics data analysis
- **Features**:
  - NPX (Normalized Protein eXpression) data handling
  - Cross-platform biomarker discovery
  - Protein-level differential expression
- **Key Components**:
  - **Phase 1**: Data structure analysis and verification
  - **Phase 2**: Statistical analysis
  - **Phase 3**: Comprehensive comparisons and visualizations

#### 5. Temporal Analysis (`Time Dependent PemIL2.py`)
- **Purpose**: Longitudinal analysis of treatment effects over time
- **Features**:
  - Time-dependent expression changes
  - Response group stratification
  - Difference-in-differences analysis
  - IL-2 treatment effect quantification
- **Key Analyses**:
  - Individual timepoint comparisons
  - Pooled IL-2 response analysis
  - Comprehensive differential response analysis

## Data Requirements

### Input Files
- **Primary Data**: `CICPT_1558_data (2).xlsx` or `CICPT_1558_data (3).xlsx`
  - Multiple sheets: `sampleID`, `nanostring`, `Olink`, `group`
- **Flow Cytometry**: `18536 PD-1 flow data pivot 250611.xlsx`
- **Response Mapping**: `ID response correlation.xlsx`

### Data Format Specifications
- **Gene Expression**: Log2-transformed values
- **Protein Expression**: NPX units (log2-scale)
- **Flow Cytometry**: % of parent population, median fluorescence intensity
- **Sample IDs**: Format like `01-027.B2W2FC`
- **Response Groups**: EXT, INT, NON classifications

## Installation and Setup

### Required Packages
```python
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
gseapy
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn gseapy
```

## Usage Examples

### Basic Flow Cytometry Analysis
```python
# Load and run complete flow cytometry analysis
from Flow_Cytometry_PemIL2 import run_flow_analysis

# Prepare data (ensure proper file path)
flow_df_ready = prepare_flow_data_for_analysis('CICPT_1558_data (3).xlsx')

# Run comprehensive analysis
analyzer, results, summary_df = run_flow_analysis(flow_df_ready)
```

### NanoString Expression Analysis
```python
# Run phase-by-phase analysis
exec(open('NanoString_PemIL2.py').read())

# Phase 1: Data verification
# Phase 2: Differential expression
# Phase 3: Visualization
```

### GSEA Pathway Analysis
```python
# Run corrected GSEA analysis
exec(open('GSEA_PemIL2.py').read())

# Generates:
# - Pathway enrichment results
# - Publication-quality plots
# - Tiered significance files
```

### Temporal Analysis
```python
# Complete temporal analysis pipeline
exec(open('Time_Dependent_PemIL2.py').read())

# Includes:
# - Individual timepoint analysis
# - IL-2 treatment effects
# - Differential response patterns
```

## Key Statistical Methods

### Differential Expression
- **Test Selection**: Mann-Whitney U test (non-parametric) or Welch's t-test
- **Multiple Testing**: Benjamini-Hochberg FDR correction
- **Effect Size**: Cohen's d, fold change calculations
- **Thresholds**: p < 0.05, FDR < 0.05, |log2FC| > 0.6

### GSEA Analysis
- **Ranking Metric**: Log2 fold change
- **Permutations**: 5000
- **Gene Sets**: GO, KEGG, Reactome, WikiPathway
- **Significance**: FDR q-value < 0.25

### Temporal Analysis
- **Paired Analysis**: Paired t-tests for longitudinal data
- **Difference-in-Differences**: Group comparison of treatment effects
- **Missing Data**: Available case analysis

## Output Files

### Generated Results
- **Flow Cytometry**: 
  - `flow_analysis_results.csv`
  - `CD16_NK_cells_main_figure.png/pdf`
- **NanoString**:
  - `nanostring_corrected_comprehensive.csv`
  - `nanostring_volcano_corrected.png/pdf`
  - `nanostring_top_genes_violin_boxplots.png/pdf`
- **GSEA**:
  - `comprehensive_gsea_summary_corrected.csv`
  - Platform-specific pathway plots
  - Tiered significance files (FDR 0.05, 0.10, 0.25)
- **Temporal**:
  - Individual timepoint volcano plots
  - Heatmaps by timepoint
  - IL-2 treatment response analysis

### Visualization Types
- **Volcano Plots**: Statistical significance vs. fold change
- **Heatmaps**: Expression patterns across groups/timepoints
- **Box/Violin Plots**: Group comparisons with significance testing
- **Pathway Plots**: GSEA enrichment results
- **Temporal Trajectories**: Longitudinal expression changes

## Data Corrections and Validation

### Log2 Transformation Handling
- **Correct Method**: `log2FC = mean_group1 - mean_group2` (for log2 data)

### Quality Control Features
- Sample mapping verification
- Control gene identification and exclusion
- Missing data pattern analysis
- Statistical assumption checking
- Cross-platform validation

## Troubleshooting

### Common Issues
1. **File Path Errors**: Ensure Excel files are in the working directory
2. **Sample Mapping**: Verify patient ID formats match between sheets
3. **Memory Issues**: Large datasets may require increased memory allocation
4. **Missing Dependencies**: Install all required packages before running

### Data Validation
- Check log2 transformation status
- Verify response group mappings
- Confirm timepoint availability
- Validate control gene exclusion

## Citation

When using this analysis pipeline, please cite the associated clinical trial:
- **ClinicalTrials.gov Identifier**: NCT02964078
- **Study Title**: Interleukin-2 and Pembrolizumab for Metastatic Kidney Cancer

## Contact

For questions regarding the analysis pipeline or data interpretation, please refer to the clinical trial documentation or contact the study investigators.

## License

This analysis code is provided for research purposes. Please ensure compliance with institutional and ethical guidelines when using clinical trial data.
