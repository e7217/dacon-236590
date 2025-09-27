# Data Model: File Organization Entities

## Overview
This document describes the entities (file types and directories) involved in the project reorganization, their attributes, relationships, and transformation rules.

## Entities

### 1. Python Script
**Description**: Executable Python files (.py) that perform specific analysis, modeling, or utility tasks

**Attributes**:
- `filename`: String - Original filename (e.g., "eda.py")
- `purpose`: Enum - [analysis, preprocessing, modeling, submission, tracking]
- `current_location`: Path - Current file path (typically root directory)
- `target_location`: Path - Destination path after reorganization
- `is_multi_purpose`: Boolean - Whether file serves multiple purposes
- `secondary_purposes`: List[String] - Additional purposes if multi-purpose

**Validation Rules**:
- Filename must end with .py
- Purpose must be determinable from filename or content analysis
- Multi-purpose files must document secondary purposes in target directory README

**State Transitions**:
1. Discovered (in root)
2. Categorized (purpose assigned)
3. Moved (to target location)
4. Validated (exists at target, executable)

**Examples**:
- `eda.py` → scripts/analysis/eda.py
- `preprocessing.py` → scripts/preprocessing/preprocessing.py
- `baseline_models.py` → scripts/modeling/baseline_models.py
- `simple_submission.py` → scripts/submissions/simple_submission.py

### 2. Jupyter Notebook
**Description**: Interactive notebook files (.ipynb) combining code, visualizations, and narrative

**Attributes**:
- `filename`: String - Original filename (e.g., "main.ipynb")
- `analysis_stage`: Integer - Workflow stage number (1-7)
- `stage_name`: String - Descriptive stage name (e.g., "eda", "preprocessing")
- `current_location`: Path - Current file path
- `target_location`: Path - notebooks/ with numbered prefix

**Validation Rules**:
- Filename must end with .ipynb
- Must be executable from top to bottom
- Stage number must be 01-99 (zero-padded)
- Stage name must reflect analysis workflow

**Naming Convention**:
```
{stage_number}_{stage_name}.ipynb
Examples: 01_eda.ipynb, 02_preprocessing.ipynb, 03_modeling.ipynb
```

**State Transitions**:
1. Discovered
2. Stage assigned
3. Renamed (if needed)
4. Moved to notebooks/
5. Validated (runnable)

### 3. Visualization Output
**Description**: PNG image files showing analysis results and visualizations

**Attributes**:
- `filename`: String - Descriptive filename (e.g., "correlation_heatmap.png")
- `category`: Enum - [eda, model_performance, feature_analysis, other]
- `current_location`: Path - Usually root directory
- `target_location`: Path - outputs/figures/ with optional subdirectory

**Validation Rules**:
- Filename must end with .png (or other image extensions)
- Filename should be descriptive of visualization content
- Must not be executable files

**Target Organization**:
```
outputs/figures/
├── eda/                          # EDA visualizations
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   └── dimensionality_analysis.png
├── feature_importance.png
└── target_distribution.png
```

### 4. Model Artifact
**Description**: Serialized models, preprocessors, and processed datasets

**Attributes**:
- `filename`: String - Artifact filename (e.g., "preprocessor_basic.pkl")
- `artifact_type`: Enum - [preprocessor, model, processed_data]
- `current_location`: Path - Root directory
- `target_location`: Path - models/ directory
- `should_gitignore`: Boolean - True for all large binary files

**Validation Rules**:
- Filename ends with .pkl or other serialization format
- Must be reproducible from scripts
- Must be added to .gitignore if size > 1MB

**Gitignore Patterns**:
```
*.pkl
models/*.pkl
data/processed/*.pkl
processed_data*.pkl
preprocessor*.pkl
```

**State Transitions**:
1. Discovered
2. Categorized by type
3. Moved to models/
4. Added to .gitignore
5. Validated (can be regenerated)

### 5. Submission File
**Description**: CSV files formatted for competition submission

**Attributes**:
- `filename`: String - Submission filename (e.g., "submission.csv")
- `version`: String - Version identifier (e.g., "baseline", "v2")
- `timestamp`: DateTime - Creation timestamp
- `current_location`: Path - Root directory
- `target_location`: Path - outputs/submissions/ with versioned name

**Validation Rules**:
- Filename must end with .csv
- Must contain required competition columns (ID, target)
- Should include version/timestamp in filename

**Naming Convention**:
```
{model_type}_{version}_{YYYYMMDD}.csv
Examples: baseline_v1_20250926.csv, ensemble_v2_20250926.csv
```

### 6. Documentation File
**Description**: Markdown files describing project, analysis, and results

**Attributes**:
- `filename`: String - Documentation filename (e.g., "README.md")
- `doc_type`: Enum - [readme, project_summary, analysis_report, other]
- `current_location`: Path - Root or answers/ directory
- `target_location`: Path - docs/ or docs/analysis_reports/

**Validation Rules**:
- Filename must end with .md
- README.md kept in root (or symlinked) for GitHub visibility
- Analysis reports maintain chronological naming

**Target Organization**:
```
docs/
├── README.md                          # Main documentation
├── PROJECT_SUMMARY.md                 # Project overview
├── analysis_reports/                  # Historical analysis
│   ├── chapter_00_current_codebase_analysis.md
│   ├── chapter3_phase1_task_T007_results.md
│   ├── master_plan_0_90_achievement.md
│   └── tasks_0_90_achievement.md
└── DIRECTORY_STRUCTURE.md             # This reorganization guide
```

### 7. Directory
**Description**: Folder structures organizing related files

**Attributes**:
- `path`: Path - Directory path
- `purpose`: String - Directory purpose description
- `contains`: List[Entity] - Files and subdirectories
- `has_readme`: Boolean - Whether directory has README.md

**Key Directories**:
```
notebooks/          # Jupyter notebooks by stage
scripts/            # Executable Python scripts
  ├── analysis/     # EDA and exploration
  ├── preprocessing/# Data transformation
  ├── modeling/     # Model training
  ├── submissions/  # Submission generation
  └── tracking/     # Experiment tracking
src/                # Reusable modules (existing)
data/               # Data files
  ├── raw/          # Original competition data
  └── processed/    # Transformed data (gitignored)
models/             # Trained models (gitignored)
outputs/            # Generated outputs
  ├── figures/      # Visualizations
  └── submissions/  # Competition submissions
docs/               # Documentation
experiments/        # MLflow, WandB (existing)
tests/              # Test files (existing)
config/             # Configuration (existing)
```

## Relationships

### Script → Directory
- **Relationship**: "belongs_to"
- **Cardinality**: Many scripts to one directory
- **Rules**: Script purpose determines target directory

### Notebook → Stage
- **Relationship**: "represents"
- **Cardinality**: One notebook to one workflow stage
- **Rules**: Stage number enforces execution order

### Script → Artifact
- **Relationship**: "generates"
- **Cardinality**: One script generates many artifacts
- **Rules**: Artifacts must be reproducible from scripts

### Notebook → Visualization
- **Relationship**: "produces"
- **Cardinality**: One notebook produces many visualizations
- **Rules**: Visualizations saved to outputs/figures/

### Documentation → Entity
- **Relationship**: "describes"
- **Cardinality**: One doc describes many entities
- **Rules**: DIRECTORY_STRUCTURE.md maps all files

## Conflict Resolution

### Filename Conflicts
When multiple files have the same name but different locations:

**Detection**: Check for duplicate filenames across source locations

**Resolution Strategy** (per clarification):
1. Flag conflict for manual review
2. Determine if files are:
   - Identical (deduplicate - keep one)
   - Different versions (merge or add version suffix)
   - Different purposes (rename with context)

**Example**:
```
# Conflict: Two "model.pkl" files
models/baseline/model.pkl
models/ensemble/model.pkl

# Resolution: Context-specific naming
models/baseline/baseline_model.pkl
models/ensemble/ensemble_model.pkl
```

### Multi-Purpose Files
Files serving multiple purposes:

**Detection**: Analyze script imports and function calls

**Resolution** (per clarification):
- Place in primary purpose directory
- Document secondary purposes in directory README

**Example**:
```python
# File: analysis_and_submission.py
# Primary: Analysis (more code devoted to EDA)
# Secondary: Also generates submission file

# Placement: scripts/analysis/analysis_and_submission.py
# Documentation: scripts/analysis/README.md mentions submission capability
```

## Validation Schema

### Post-Move Validation Checklist
For each entity type, verify:

1. **Scripts**:
   - [ ] All .py files categorized and moved
   - [ ] Executable permissions preserved
   - [ ] Import paths still work (relative → absolute if needed)

2. **Notebooks**:
   - [ ] Numbered sequentially (01, 02, 03, ...)
   - [ ] Runnable from top to bottom
   - [ ] Kernel paths still valid

3. **Outputs**:
   - [ ] All PNG files in outputs/figures/
   - [ ] All CSV submissions in outputs/submissions/
   - [ ] Large binaries gitignored

4. **Documentation**:
   - [ ] README accessible (root or symlink)
   - [ ] All analysis reports in docs/analysis_reports/
   - [ ] New DIRECTORY_STRUCTURE.md created

5. **Integrity**:
   - [ ] No files lost (count matches)
   - [ ] Git history preserved
   - [ ] No broken symlinks

## Implementation Notes

This data model supports the reorganization implementation by:
- Defining clear entity types and purposes
- Establishing validation rules for each entity
- Specifying conflict resolution strategies
- Providing relationship mappings for verification

All entities align with constitutional principles:
- Jupyter Notebook-First (notebooks/ prioritized)
- Educational Clarity (clear directory purposes)
- Performance-First (no impact on model performance)
- Iterative Reflection (docs/ for analysis reports)