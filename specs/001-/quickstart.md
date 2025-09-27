# Quickstart: File Reorganization Validation

## Purpose
This guide provides step-by-step validation procedures to verify the project file reorganization was completed successfully and meets all functional requirements from spec.md.

## Prerequisites
- Git repository with branch `001-` checked out
- Python 3.11+ environment
- Read access to all project directories

## Validation Steps

### Step 1: Verify Directory Structure Exists
**Expected Duration**: 30 seconds

```bash
# Navigate to project root
cd /home/e7217/projects/dacon-236590

# Check all required directories exist
ls -la | grep -E "notebooks|scripts|docs|outputs|models"

# Check scripts subdirectories
ls scripts/

# Check outputs subdirectories
ls outputs/

# Expected output:
# notebooks/
# scripts/ (with analysis/, preprocessing/, modeling/, submissions/, tracking/)
# docs/
# outputs/ (with figures/, submissions/)
# models/
```

**Success Criteria**:
- [ ] notebooks/ directory exists
- [ ] scripts/ with 5 subdirectories exists
- [ ] docs/ directory exists
- [ ] outputs/figures/ exists
- [ ] outputs/submissions/ exists
- [ ] models/ directory exists
- [ ] src/ directory still exists (unchanged)
- [ ] experiments/ directory still exists (unchanged)

### Step 2: Validate Python Scripts Organization
**Expected Duration**: 2 minutes

```bash
# Count files in each scripts subdirectory
echo "Analysis scripts:"
ls scripts/analysis/*.py

echo "Preprocessing scripts:"
ls scripts/preprocessing/*.py

echo "Modeling scripts:"
ls scripts/modeling/*.py

echo "Submission scripts:"
ls scripts/submissions/*.py

echo "Tracking scripts:"
ls scripts/tracking/*.py
```

**Success Criteria** (FR-001):
- [ ] eda.py in scripts/analysis/
- [ ] advanced_analysis.py in scripts/analysis/
- [ ] preprocessing.py in scripts/preprocessing/
- [ ] baseline_models.py in scripts/modeling/
- [ ] advanced_ensemble.py in scripts/modeling/
- [ ] final_solution.py in scripts/modeling/
- [ ] simple_submission.py in scripts/submissions/
- [ ] quick_baseline.py in scripts/submissions/
- [ ] setup_experiment_tracking.py in scripts/tracking/
- [ ] view_experiments.py in scripts/tracking/
- [ ] test_libraries.py in scripts/tracking/

### Step 3: Validate Jupyter Notebooks Organization
**Expected Duration**: 1 minute

```bash
# List notebooks with naming convention
ls -la notebooks/

# Check for README
cat notebooks/README.md
```

**Success Criteria** (FR-002):
- [ ] main.ipynb renamed to numbered format (e.g., 01_eda.ipynb or similar)
- [ ] Notebooks have stage-based naming (01_*, 02_*, etc.)
- [ ] notebooks/README.md exists explaining execution order
- [ ] All .ipynb files are in notebooks/ directory (none in root)

### Step 4: Validate Data Organization
**Expected Duration**: 1 minute

```bash
# Check data directory structure
ls -la data/
ls -la data/raw/ 2>/dev/null || echo "data/raw/ may not exist yet"
ls -la data/processed/ 2>/dev/null || echo "data/processed/ may not exist yet"

# Check models directory
ls -la models/ 2>/dev/null || echo "models/ may not exist yet"

# Verify .gitignore patterns
grep -E "\.pkl|processed_data|preprocessor" .gitignore
```

**Success Criteria** (FR-003):
- [ ] data/ directory exists
- [ ] data/raw/ exists (or data/open/ kept as is)
- [ ] data/processed/ directory created (may be empty)
- [ ] models/ directory exists
- [ ] .gitignore includes *.pkl pattern
- [ ] .gitignore includes processed_data* pattern
- [ ] .gitignore includes preprocessor* pattern
- [ ] Large .pkl files NOT in root directory

### Step 5: Validate Visualization Organization
**Expected Duration**: 1 minute

```bash
# List visualization files
ls -la outputs/figures/

# Count PNG files
echo "Total PNG files:"
find outputs/figures/ -name "*.png" | wc -l

# Expected files (at least):
# - class_distribution.png
# - correlation_heatmap.png
# - dimensionality_analysis.png
# - feature_importance.png
# - target_distribution.png
```

**Success Criteria** (FR-004):
- [ ] outputs/figures/ directory exists
- [ ] class_distribution.png moved to outputs/figures/
- [ ] correlation_heatmap.png moved to outputs/figures/
- [ ] dimensionality_analysis.png moved to outputs/figures/
- [ ] feature_importance.png moved to outputs/figures/
- [ ] target_distribution.png moved to outputs/figures/
- [ ] No PNG files remain in root directory

### Step 6: Validate Documentation Organization
**Expected Duration**: 2 minutes

```bash
# Check docs directory
ls -la docs/

# Check analysis reports subdirectory
ls -la docs/analysis_reports/

# Verify main documentation files
ls docs/ | grep -E "README|PROJECT_SUMMARY|DIRECTORY_STRUCTURE"
```

**Success Criteria** (FR-005, FR-011):
- [ ] docs/ directory exists
- [ ] README.md accessible (in docs/ or symlinked from root)
- [ ] PROJECT_SUMMARY.md in docs/
- [ ] docs/analysis_reports/ directory exists
- [ ] Files from answers/ moved to docs/analysis_reports/
- [ ] DIRECTORY_STRUCTURE.md created documenting new organization

### Step 7: Validate Submission Files Organization
**Expected Duration**: 30 seconds

```bash
# List submission files
ls -la outputs/submissions/

# Check for CSV files
find outputs/submissions/ -name "*.csv"
```

**Success Criteria** (FR-006):
- [ ] outputs/submissions/ directory exists
- [ ] submission.csv moved to outputs/submissions/
- [ ] final_submission.csv moved to outputs/submissions/
- [ ] Filenames include version/timestamp (or documented in README)
- [ ] No CSV submission files in root directory

### Step 8: Validate src/ Directory Unchanged
**Expected Duration**: 30 seconds

```bash
# Verify src/ structure preserved
ls -la src/

# Check for existing subdirectories
ls src/ | grep -E "analysis|diagnosis|tracking|utils"
```

**Success Criteria** (FR-007):
- [ ] src/ directory structure unchanged
- [ ] src/analysis/ exists
- [ ] src/diagnosis/ exists
- [ ] src/tracking/ exists
- [ ] src/utils/ exists
- [ ] No script files (.py with if __name__ == "__main__") in src/

### Step 9: Validate Experiments Directory Unchanged
**Expected Duration**: 30 seconds

```bash
# Verify experiments/ structure preserved
ls -la experiments/

# Check for tracking outputs
ls experiments/ | grep -E "mlruns|wandb|plots"
```

**Success Criteria** (FR-008):
- [ ] experiments/ directory unchanged
- [ ] experiments/mlruns/ exists
- [ ] experiments/wandb/ exists
- [ ] experiments/plots/ exists

### Step 10: Validate Configuration Files Preserved
**Expected Duration**: 30 seconds

```bash
# Check root configuration files
ls -la | grep -E "pyproject.toml|.gitignore|.python-version"

# Verify they are still in root
pwd
ls pyproject.toml .gitignore .python-version
```

**Success Criteria** (FR-010):
- [ ] pyproject.toml in root directory
- [ ] .gitignore in root directory (with updates)
- [ ] .python-version in root directory
- [ ] No configuration files moved from root

### Step 11: File Integrity Check
**Expected Duration**: 2 minutes

```bash
# Count files before reorganization (baseline)
# Use git to compare file counts

# Check for file loss
find . -name "*.py" | grep -v ".venv" | grep -v ".git" | wc -l
find . -name "*.ipynb" | grep -v ".venv" | grep -v ".git" | wc -l
find . -name "*.png" | grep -v ".venv" | grep -v ".git" | wc -l
find . -name "*.csv" | grep -v "data/" | grep -v ".venv" | wc -l

# Verify git status
git status
```

**Success Criteria**:
- [ ] File count matches expected (no files lost)
- [ ] Git shows moves, not deletes + adds (preserves history)
- [ ] No untracked files that should be tracked
- [ ] No accidentally committed large binary files

### Step 12: Functional Verification
**Expected Duration**: 5 minutes

```bash
# Test that scripts still execute
# (Choose one or two key scripts to verify)

cd /home/e7217/projects/dacon-236590

# Test analysis script
python scripts/analysis/eda.py --help  # Check if script is executable

# Test notebook can be opened
# jupyter notebook notebooks/01_eda.ipynb  # Manual check

# Test import paths (if any scripts import from src/)
python -c "import sys; sys.path.insert(0, '.'); from src.utils import *"  # Check imports work
```

**Success Criteria**:
- [ ] Scripts are executable (no permission issues)
- [ ] Import paths still work (no broken imports)
- [ ] Notebooks can be opened in Jupyter
- [ ] No critical functionality broken

### Step 13: Documentation Verification
**Expected Duration**: 3 minutes

```bash
# Read DIRECTORY_STRUCTURE.md
cat docs/DIRECTORY_STRUCTURE.md

# Verify it documents:
# - All major directories
# - Purpose of each directory
# - File organization rationale
```

**Success Criteria** (FR-011):
- [ ] DIRECTORY_STRUCTURE.md exists
- [ ] Contains directory tree diagram
- [ ] Explains purpose of each directory
- [ ] Documents file movement decisions
- [ ] Includes multi-purpose file documentation (if any)

### Step 14: Conflict Resolution Verification
**Expected Duration**: 2 minutes

**Manual Check**:
1. Review any flagged filename conflicts (if documented)
2. Verify resolution strategy was applied consistently
3. Check for duplicate files in multiple locations

**Success Criteria** (FR-012):
- [ ] No unresolved filename conflicts
- [ ] Duplicate files appropriately handled
- [ ] Multi-purpose files documented in README
- [ ] Conflict resolution documented in DIRECTORY_STRUCTURE.md

## Acceptance Test Results

### Test Summary
| Step | Requirement | Status | Notes |
|------|-------------|--------|-------|
| 1 | Directory Structure | ⬜ | |
| 2 | Python Scripts (FR-001) | ⬜ | |
| 3 | Notebooks (FR-002) | ⬜ | |
| 4 | Data/Models (FR-003) | ⬜ | |
| 5 | Visualizations (FR-004) | ⬜ | |
| 6 | Documentation (FR-005) | ⬜ | |
| 7 | Submissions (FR-006) | ⬜ | |
| 8 | src/ Preserved (FR-007) | ⬜ | |
| 9 | Experiments Preserved (FR-008) | ⬜ | |
| 10 | Config Preserved (FR-010) | ⬜ | |
| 11 | File Integrity | ⬜ | |
| 12 | Functional Test | ⬜ | |
| 13 | Documentation (FR-011) | ⬜ | |
| 14 | Conflicts (FR-012) | ⬜ | |

### Overall Status
- [ ] All critical requirements (FR-001 through FR-012) verified
- [ ] No files lost or corrupted
- [ ] Git history preserved
- [ ] Scripts and notebooks functional
- [ ] Documentation complete

## Rollback Procedure
If validation fails:

```bash
# Check out previous commit (before reorganization)
git log --oneline | head -5
git checkout <commit-hash-before-reorganization>

# Or if on branch:
git checkout main
git branch -D 001-
```

## Success Confirmation
Once all checkboxes are marked:
1. Commit changes: `git commit -m "docs: complete file reorganization per spec 001"`
2. Merge to main (if applicable): `git checkout main && git merge 001-`
3. Update PROJECT_SUMMARY.md with new structure
4. Close specification: Mark spec.md as Status: Complete

## Notes for Implementation Team
- Preserve git history by using `git mv` instead of `mv`
- Create directories before moving files
- Update .gitignore BEFORE moving large binary files
- Document any deviations from plan in DIRECTORY_STRUCTURE.md