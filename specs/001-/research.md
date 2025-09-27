# Research: Project File Organization Best Practices

## Research Questions
1. What are best practices for organizing data science project directories?
2. How should we handle large binary files in Git repositories?
3. What directory structure supports Jupyter notebook-first workflows?
4. How to organize scripts vs. modules vs. notebooks?

## Findings

### 1. Data Science Project Structure Best Practices

**Decision**: Use Cookiecutter Data Science inspired structure with modifications for competition context

**Rationale**:
- Industry standard: Cookiecutter Data Science is widely adopted
- Clear separation of concerns: data, notebooks, src (modules), outputs
- Supports reproducibility: Clear paths for raw vs. processed data
- Competition friendly: Accommodates submissions, experiments, and tracking

**Alternatives Considered**:
- Flat structure: Rejected - doesn't scale, hard to navigate
- Deep nesting: Rejected - overcomplicates for single-person project
- Domain-driven design: Rejected - not applicable to analysis workflow

**References**:
- Cookiecutter Data Science: https://drivendata.github.io/cookiecutter-data-science/
- Kaggle Best Practices: Community standards for competition projects

### 2. Large Binary File Management

**Decision**: Add *.pkl, processed_data*, preprocessor* patterns to .gitignore

**Rationale**:
- Repository bloat prevention: Binary files grow Git repo significantly
- Regenerable artifacts: Models and processed data can be recreated from scripts
- Version control focus: Track code and configs, not outputs
- Clarification confirmed: User selected option A (gitignore, local generation only)

**Alternatives Considered**:
- Git LFS: Rejected - adds complexity, cost for private repos
- External storage: Rejected - unnecessary for local competition work
- Commit everything: Rejected - causes repo bloat, slow clones

**Implementation**:
```gitignore
# Models and processed data
*.pkl
**/processed_data*.pkl
**/preprocessor*.pkl
models/*.pkl
data/processed/*.pkl
```

### 3. Jupyter Notebook Organization

**Decision**: notebooks/ directory with numbered prefixes indicating workflow stage

**Rationale**:
- Constitutional alignment: Supports Principle I (Jupyter Notebook-First Development)
- Clear progression: Numbers (01_, 02_, 03_) show analysis order
- Educational value: Easy for others to follow analysis workflow
- Stage-based: Matches EDA → Preprocessing → Modeling → Evaluation workflow

**Naming Convention**:
```
notebooks/
├── 01_eda.ipynb                    # Exploratory Data Analysis
├── 02_feature_engineering.ipynb   # Feature creation and selection
├── 03_preprocessing.ipynb         # Data cleaning and transformation
├── 04_baseline_modeling.ipynb     # Simple benchmark models
├── 05_advanced_modeling.ipynb     # Sophisticated algorithms
├── 06_ensemble.ipynb              # Model combination
├── 07_evaluation.ipynb            # Results analysis
└── README.md                       # Notebook execution order and purpose
```

**Alternatives Considered**:
- Date-based naming: Rejected - doesn't show logical progression
- Descriptive only: Rejected - unclear execution order
- Flat in root: Rejected - clutter, no organization

### 4. Scripts vs. Modules vs. Notebooks

**Decision**:
- **scripts/**: Executable Python files organized by purpose subdirectories
- **src/**: Reusable modules and libraries
- **notebooks/**: Interactive analysis and documentation

**Rationale**:
- Clear purpose distinction: Scripts execute workflows, src provides utilities
- Subdirectory organization: scripts/analysis/, scripts/modeling/, scripts/submissions/
- Multi-purpose handling: Place in primary purpose directory, document in README (per clarification)
- Module reuse: src/ contains importable code used by both scripts and notebooks

**Directory Purposes**:
```
scripts/               # Executable workflows
├── analysis/         # EDA and visualization scripts
├── preprocessing/    # Data transformation scripts
├── modeling/         # Model training scripts
├── submissions/      # Competition submission generators
└── tracking/         # Experiment tracking utilities

src/                  # Importable modules
├── analysis/         # Analysis utilities
├── diagnosis/        # Debugging and diagnostics
├── tracking/         # Tracking helpers
└── utils/            # General utilities
```

**Alternatives Considered**:
- Everything in src/: Rejected - mixes scripts and modules
- No src/ directory: Rejected - code duplication across scripts
- Flat scripts/: Rejected - too many files, no categorization

### 5. Output Organization

**Decision**:
- outputs/figures/ for visualizations (PNG files)
- outputs/submissions/ for competition CSV files
- models/ for trained model artifacts

**Rationale**:
- Clarification confirmed: User selected outputs/figures/ structure
- Logical grouping: All outputs under outputs/ parent directory
- Type separation: figures/ vs. submissions/ vs. models/
- Gitignore friendly: Can exclude entire outputs/ and models/ if needed

**Structure**:
```
outputs/
├── figures/                    # All PNG visualizations
│   ├── eda/                   # EDA plots
│   ├── model_performance/     # Model evaluation plots
│   └── feature_importance/    # Feature analysis plots
└── submissions/               # Competition submissions
    ├── baseline_20250926.csv
    └── ensemble_v2_20250926.csv

models/                        # Trained models (gitignored)
├── baseline/
├── ensemble/
└── final/
```

### 6. Documentation Consolidation

**Decision**: docs/ directory for all documentation with subdirectories for analysis reports

**Rationale**:
- Centralized knowledge: One place for all project documentation
- README accessibility: Main README moves to docs/ with symlink in root (optional)
- Analysis reports: answers/ directory content moves to docs/analysis_reports/
- Historical preservation: Maintains chronological analysis progression

**Structure**:
```
docs/
├── README.md                          # Main project documentation
├── PROJECT_SUMMARY.md                 # Comprehensive project overview
├── analysis_reports/                  # From answers/ directory
│   ├── chapter_00_current_codebase_analysis.md
│   ├── master_plan_0_90_achievement.md
│   └── tasks_0_90_achievement.md
└── DIRECTORY_STRUCTURE.md             # This reorganization documentation
```

## Summary

The reorganization follows industry best practices while adapting to:
1. **Competition context**: submissions/, experiments/ directories
2. **Constitutional principles**: notebooks/ first, clear educational structure
3. **User clarifications**: outputs/figures/, gitignore for binaries, primary purpose placement
4. **Workflow support**: Stage-based organization (EDA → Modeling → Evaluation)

All decisions are testable through acceptance criteria in spec.md and support the Jupyter notebook-first, educational, and performance-focused constitutional principles.