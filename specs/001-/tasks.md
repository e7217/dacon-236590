# Tasks: Project File Organization and Categorization

**Input**: Design documents from `/home/e7217/projects/dacon-236590/specs/001-/`
**Prerequisites**: plan.md (required), research.md, data-model.md, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extract: target directory structure, file mappings
2. Load data-model.md:
   → Extract entities: Script (11 files), Notebook (1 file), Visualization (5 files),
     Model Artifact (6 files), Submission (2 files), Documentation (3+ files)
3. Load quickstart.md: 14 validation steps
4. Generate tasks by category:
   → Setup: Directory creation, .gitignore preparation
   → Move: Categorize and relocate files using git mv
   → Documentation: Create READMEs, generate structure documentation
   → Validation: Execute quickstart.md checks
5. Apply task rules:
   → Different files = mark [P] for parallel execution
   → Same directory READMEs = sequential (dependencies)
   → Use git mv to preserve history
6. Number tasks sequentially (T001, T002...)
7. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- Repository root: `/home/e7217/projects/dacon-236590/`
- All paths are absolute or relative to repository root

---

## Phase 3.1: Setup & Preparation

- [X] T001 Create target directory structure
  - Create: `notebooks/`, `scripts/analysis/`, `scripts/preprocessing/`, `scripts/modeling/`, `scripts/submissions/`, `scripts/tracking/`, `outputs/figures/`, `outputs/submissions/`, `models/`, `docs/`, `docs/analysis_reports/`
  - Verify: All directories created successfully
  - Command: `mkdir -p notebooks scripts/{analysis,preprocessing,modeling,submissions,tracking} outputs/{figures,submissions} models docs/analysis_reports`

- [X] T002 Update .gitignore for large binary files
  - File: `.gitignore`
  - Add patterns: `*.pkl`, `**/processed_data*.pkl`, `**/preprocessor*.pkl`, `models/*.pkl`, `data/processed/*.pkl`
  - Verify: Patterns added without duplicating existing entries
  - Purpose: Prevent repository bloat from model artifacts

- [X] T003 Create backup reference of current file locations
  - Command: `find . -maxdepth 1 -type f \( -name "*.py" -o -name "*.ipynb" -o -name "*.png" -o -name "*.csv" -o -name "*.pkl" \) > docs/file_inventory_before_reorganization.txt`
  - Purpose: Reference for validation and rollback if needed

---

## Phase 3.2: Move Python Scripts (Using git mv)

### Analysis Scripts
- [X] T004 [P] Move eda.py to scripts/analysis/
  - Command: `git mv eda.py scripts/analysis/eda.py`
  - Verify: File exists at scripts/analysis/eda.py, executable

- [X] T005 [P] Move advanced_analysis.py to scripts/analysis/
  - Command: `git mv advanced_analysis.py scripts/analysis/advanced_analysis.py`
  - Verify: File exists at scripts/analysis/advanced_analysis.py, executable

### Preprocessing Scripts
- [X] T006 Move preprocessing.py to scripts/preprocessing/
  - Command: `git mv preprocessing.py scripts/preprocessing/preprocessing.py`
  - Verify: File exists at scripts/preprocessing/preprocessing.py, executable

### Modeling Scripts
- [X] T007 [P] Move baseline_models.py to scripts/modeling/
  - Command: `git mv baseline_models.py scripts/modeling/baseline_models.py`
  - Verify: File exists at scripts/modeling/baseline_models.py, executable

- [X] T008 [P] Move advanced_ensemble.py to scripts/modeling/
  - Command: `git mv advanced_ensemble.py scripts/modeling/advanced_ensemble.py`
  - Verify: File exists at scripts/modeling/advanced_ensemble.py, executable

- [X] T009 [P] Move final_solution.py to scripts/modeling/
  - Command: `git mv final_solution.py scripts/modeling/final_solution.py`
  - Verify: File exists at scripts/modeling/final_solution.py, executable

### Submission Scripts
- [X] T010 [P] Move simple_submission.py to scripts/submissions/
  - Command: `git mv simple_submission.py scripts/submissions/simple_submission.py`
  - Verify: File exists at scripts/submissions/simple_submission.py, executable

- [X] T011 [P] Move quick_baseline.py to scripts/submissions/
  - Command: `git mv quick_baseline.py scripts/submissions/quick_baseline.py`
  - Verify: File exists at scripts/submissions/quick_baseline.py, executable

### Tracking Scripts
- [X] T012 [P] Move setup_experiment_tracking.py to scripts/tracking/
  - Command: `git mv setup_experiment_tracking.py scripts/tracking/setup_experiment_tracking.py`
  - Verify: File exists at scripts/tracking/setup_experiment_tracking.py, executable

- [X] T013 [P] Move view_experiments.py to scripts/tracking/
  - Command: `git mv view_experiments.py scripts/tracking/view_experiments.py`
  - Verify: File exists at scripts/tracking/view_experiments.py, executable

- [X] T014 [P] Move test_libraries.py to scripts/tracking/
  - Command: `git mv test_libraries.py scripts/tracking/test_libraries.py`
  - Verify: File exists at scripts/tracking/test_libraries.py, executable

---

## Phase 3.3: Move Jupyter Notebooks

- [X] T015 Move and rename main.ipynb to notebooks/01_main_analysis.ipynb
  - Command: `git mv main.ipynb notebooks/01_main_analysis.ipynb`
  - Note: Renamed to follow stage-based naming convention (01_ prefix)
  - Verify: File exists at notebooks/01_main_analysis.ipynb, can be opened in Jupyter
  - Future: Consider splitting into stage-specific notebooks (01_eda, 02_preprocessing, etc.)

---

## Phase 3.4: Move Visualizations

- [X] T016 [P] Move class_distribution.png to outputs/figures/
  - Command: `git mv class_distribution.png outputs/figures/class_distribution.png`
  - Verify: File exists at outputs/figures/class_distribution.png

- [X] T017 [P] Move correlation_heatmap.png to outputs/figures/
  - Command: `git mv correlation_heatmap.png outputs/figures/correlation_heatmap.png`
  - Verify: File exists at outputs/figures/correlation_heatmap.png

- [X] T018 [P] Move dimensionality_analysis.png to outputs/figures/
  - Command: `git mv dimensionality_analysis.png outputs/figures/dimensionality_analysis.png`
  - Verify: File exists at outputs/figures/dimensionality_analysis.png

- [X] T019 [P] Move feature_importance.png to outputs/figures/
  - Command: `git mv feature_importance.png outputs/figures/feature_importance.png`
  - Verify: File exists at outputs/figures/feature_importance.png

- [X] T020 [P] Move target_distribution.png to outputs/figures/
  - Command: `git mv target_distribution.png outputs/figures/target_distribution.png`
  - Verify: File exists at outputs/figures/target_distribution.png

---

## Phase 3.5: Move Model Artifacts

- [X] T021 [P] Move preprocessor_basic.pkl to models/
  - Command: `git mv preprocessor_basic.pkl models/preprocessor_basic.pkl`
  - Verify: File exists at models/preprocessor_basic.pkl
  - Note: Will be gitignored after this commit

- [X] T022 [P] Move preprocessor_aggressive.pkl to models/
  - Command: `git mv preprocessor_aggressive.pkl models/preprocessor_aggressive.pkl`
  - Verify: File exists at models/preprocessor_aggressive.pkl

- [X] T023 [P] Move preprocessor_feature_selected.pkl to models/
  - Command: `git mv preprocessor_feature_selected.pkl models/preprocessor_feature_selected.pkl`
  - Verify: File exists at models/preprocessor_feature_selected.pkl

- [X] T024 [P] Move processed_data_basic.pkl to models/
  - Command: `git mv processed_data_basic.pkl models/processed_data_basic.pkl`
  - Verify: File exists at models/processed_data_basic.pkl

- [X] T025 [P] Move processed_data_aggressive.pkl to models/
  - Command: `git mv processed_data_aggressive.pkl models/processed_data_aggressive.pkl`
  - Verify: File exists at models/processed_data_aggressive.pkl

- [X] T026 [P] Move processed_data_feature_selected.pkl to models/
  - Command: `git mv processed_data_feature_selected.pkl models/processed_data_feature_selected.pkl`
  - Verify: File exists at models/processed_data_feature_selected.pkl

---

## Phase 3.6: Move Submission Files

- [X] T027 [P] Move submission.csv to outputs/submissions/
  - Command: `git mv submission.csv outputs/submissions/submission_baseline.csv`
  - Note: Renamed to include version identifier
  - Verify: File exists at outputs/submissions/submission_baseline.csv

- [X] T028 [P] Move final_submission.csv to outputs/submissions/
  - Command: `git mv final_submission.csv outputs/submissions/final_submission.csv`
  - Verify: File exists at outputs/submissions/final_submission.csv

---

## Phase 3.7: Move Documentation

- [X] T029 Move PROJECT_SUMMARY.md to docs/
  - Command: `git mv PROJECT_SUMMARY.md docs/PROJECT_SUMMARY.md`
  - Verify: File exists at docs/PROJECT_SUMMARY.md

- [X] T030 Copy README.md to docs/ (keep in root for GitHub)
  - Command: `cp README.md docs/README.md`
  - Note: README.md kept in root for GitHub visibility, copied to docs/ for documentation centralization
  - Verify: Both root/README.md and docs/README.md exist

- [X] T031 Move analysis reports from answers/ to docs/analysis_reports/
  - Command: `git mv answers/*.md docs/analysis_reports/`
  - Files to move:
    - chapter_00_current_codebase_analysis.md
    - chapter3_phase1_task_T007_results.md
    - master_plan_0_90_achievement.md
    - tasks_0_90_achievement.md
  - Verify: All files moved, answers/ directory can be removed if empty

---

## Phase 3.8: Documentation Generation

- [X] T032 Create notebooks/README.md
  - File: `notebooks/README.md`
  - Content: Explain notebook execution order, stage-based naming convention
  - Include: "01_ prefix indicates EDA stage, run notebooks in numerical order"
  - Purpose: Guide users through analysis workflow

- [X] T033 [P] Create scripts/analysis/README.md
  - File: `scripts/analysis/README.md`
  - Content: Describe analysis scripts, their purposes
  - Document any multi-purpose scripts and secondary purposes

- [X] T034 [P] Create scripts/preprocessing/README.md
  - File: `scripts/preprocessing/README.md`
  - Content: Describe preprocessing workflow, data transformation steps

- [X] T035 [P] Create scripts/modeling/README.md
  - File: `scripts/modeling/README.md`
  - Content: Describe modeling scripts (baseline, advanced, ensemble, final)
  - Include: Model types, expected outputs, performance notes

- [X] T036 [P] Create scripts/submissions/README.md
  - File: `scripts/submissions/README.md`
  - Content: Describe submission generation scripts and CSV format requirements

- [X] T037 [P] Create scripts/tracking/README.md
  - File: `scripts/tracking/README.md`
  - Content: Describe experiment tracking setup and viewing tools

- [X] T038 Generate docs/DIRECTORY_STRUCTURE.md
  - File: `docs/DIRECTORY_STRUCTURE.md`
  - Content:
    - Complete directory tree with all subdirectories
    - Purpose of each directory
    - File organization rationale
    - Constitutional alignment notes (Jupyter notebook-first, educational clarity)
    - Multi-purpose file handling decisions
    - Gitignore patterns explanation
  - Purpose: Comprehensive documentation of reorganization

---

## Phase 3.9: Validation

- [X] T039 Execute quickstart.md validation checklist
  - Run all 14 validation steps from `specs/001-/quickstart.md`
  - Key validations:
    1. Directory structure exists (Step 1)
    2. All Python scripts moved correctly (Step 2)
    3. Notebooks organized with numbering (Step 3)
    4. Data directories structured (Step 4)
    5. Visualizations in outputs/figures/ (Step 5)
    6. Documentation in docs/ (Step 6)
    7. Submissions in outputs/submissions/ (Step 7)
    8. src/ unchanged (Step 8)
    9. experiments/ unchanged (Step 9)
    10. Config files in root (Step 10)
    11. File count matches (Step 11)
    12. Scripts still executable (Step 12)
    13. DIRECTORY_STRUCTURE.md complete (Step 13)
    14. No unresolved conflicts (Step 14)
  - Output: Mark all checkboxes in quickstart.md test summary table
  - Success criteria: All checkboxes marked, no files lost

---

## Dependencies

**Sequential Dependencies**:
- T001 (directory creation) must complete before all move tasks (T004-T031)
- T002 (.gitignore update) should complete before moving .pkl files (T021-T026)
- T003 (backup) should complete before any moves
- All move tasks (T004-T031) must complete before documentation generation (T032-T038)
- Documentation generation must complete before validation (T039)

**Parallel Execution Groups**:
```
Group 1 - Analysis Scripts (can run in parallel):
- T004: Move eda.py
- T005: Move advanced_analysis.py

Group 2 - Modeling Scripts (can run in parallel):
- T007: Move baseline_models.py
- T008: Move advanced_ensemble.py
- T009: Move final_solution.py

Group 3 - Submission Scripts (can run in parallel):
- T010: Move simple_submission.py
- T011: Move quick_baseline.py

Group 4 - Tracking Scripts (can run in parallel):
- T012: Move setup_experiment_tracking.py
- T013: Move view_experiments.py
- T014: Move test_libraries.py

Group 5 - Visualizations (can run in parallel):
- T016-T020: All PNG file moves

Group 6 - Model Artifacts (can run in parallel):
- T021-T026: All .pkl file moves

Group 7 - Submissions (can run in parallel):
- T027-T028: All CSV file moves

Group 8 - README Creation (can run in parallel):
- T033-T037: All scripts/*/ README files
```

---

## Parallel Example

```bash
# Example: Launch Group 2 (Modeling Scripts) together
git mv baseline_models.py scripts/modeling/baseline_models.py &
git mv advanced_ensemble.py scripts/modeling/advanced_ensemble.py &
git mv final_solution.py scripts/modeling/final_solution.py &
wait

# Example: Launch Group 5 (Visualizations) together
git mv class_distribution.png outputs/figures/class_distribution.png &
git mv correlation_heatmap.png outputs/figures/correlation_heatmap.png &
git mv dimensionality_analysis.png outputs/figures/dimensionality_analysis.png &
git mv feature_importance.png outputs/figures/feature_importance.png &
git mv target_distribution.png outputs/figures/target_distribution.png &
wait
```

---

## Notes

- Use `git mv` instead of `mv` to preserve Git history
- Verify executable permissions after moving scripts
- Run `git status` after each phase to verify changes
- All .pkl files will be gitignored after this reorganization
- README.md remains in root for GitHub visibility (copied to docs/)
- src/, experiments/, tests/, config/ directories remain unchanged
- No files should remain in root except configuration files and symlinks

---

## Task Generation Rules

*Applied during main() execution*

1. **From Data Model Entities**:
   - Each Script entity (11 files) → individual move task [P]
   - Each Notebook entity (1 file) → rename and move task
   - Each Visualization entity (5 files) → move task [P]
   - Each Model Artifact entity (6 files) → move task [P]
   - Each Submission entity (2 files) → move task [P]
   - Each Documentation entity (3+ files) → move task

2. **From Quickstart Validation Steps**:
   - Each validation step → verification checkpoint in T039

3. **From Target Structure**:
   - Each target directory → creation in T001
   - Each directory type → README creation task [P]

4. **Ordering**:
   - Setup (T001-T003) → Moves (T004-T031) → Documentation (T032-T038) → Validation (T039)

---

## Validation Checklist

*GATE: Checked after T039 execution*

- [X] All Python scripts categorized and moved (11 files)
- [X] All notebooks organized with numbering (1 file)
- [X] All visualizations in outputs/figures/ (5 files)
- [X] All model artifacts in models/ (6 files)
- [X] All submissions in outputs/submissions/ (2 files)
- [X] All documentation in docs/ (3+ files)
- [X] Directory READMEs created (7 files)
- [X] DIRECTORY_STRUCTURE.md complete
- [X] Git history preserved (used git mv)
- [X] No files lost (count matches backup)
- [X] .gitignore updated with binary patterns
- [X] Quickstart validation passed all 14 steps

---

**Constitutional Compliance for Analysis Tasks:**
- Each task contributes to notebook-first workflow (Principle I): Creates notebooks/ structure
- Documentation tasks include educational explanations (Principle II): All READMEs explain purposes
- No impact on modeling performance (Principle III): File organization only
- Structure accommodates research artifacts (Principle IV): experiments/ preserved
- docs/ consolidates reflection materials (Principle V): analysis_reports/ subdirectory