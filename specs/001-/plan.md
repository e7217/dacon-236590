
# Implementation Plan: Project File Organization and Categorization

**Branch**: `001-` | **Date**: 2025-09-26 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/home/e7217/projects/dacon-236590/specs/001-/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Organize scattered project files into a logical directory structure that aligns with the data analysis workflow (EDA → Preprocessing → Modeling → Evaluation). Categorize Python scripts by purpose (analysis, preprocessing, modeling, submission), organize notebooks with stage-based naming, separate outputs (visualizations, submissions, models) from source code, and consolidate documentation. This reorganization supports the constitutional principle of Jupyter notebook-first development with clear educational and iterative analysis workflows.

## Technical Context
**Language/Version**: Python 3.11+ (existing project)
**Primary Dependencies**: File system operations (os, shutil, pathlib), Git (version control)
**Storage**: Local filesystem - reorganizing existing files
**Testing**: Manual validation checklist (verify all files moved correctly)
**Target Platform**: Linux development environment
**Project Type**: Data analysis project - single root with organized subdirectories
**Performance Goals**: N/A - file organization task
**Constraints**: Preserve file integrity, maintain Git history, ensure reproducibility
**Scale/Scope**: ~35 files in root directory to reorganize into ~10 logical directories

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Constitutional Principles Compliance:**
- [x] **Jupyter Notebook-First**: Reorganization creates notebooks/ directory with stage-based naming (01_eda.ipynb, 02_preprocessing.ipynb, etc.)
- [x] **Educational Clarity**: Directory structure and documentation will be clear and self-explanatory
- [x] **Performance-First**: Organization supports efficient workflow (no impact on model performance)
- [x] **Research-Driven**: Structure accommodates experiments/ directory for research artifacts
- [x] **Iterative Reflection**: docs/ directory will consolidate analysis reports and reflections

*Note: This is a file organization task that SUPPORTS constitutional principles by creating proper structure for analysis work. No constitutional violations.*

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)

**Target Structure** (after reorganization):
```
dacon-236590/
├── notebooks/                    # Jupyter notebooks organized by stage
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── README.md
├── scripts/                      # Python scripts organized by purpose
│   ├── analysis/
│   │   ├── eda.py
│   │   └── advanced_analysis.py
│   ├── preprocessing/
│   │   └── preprocessing.py
│   ├── modeling/
│   │   ├── baseline_models.py
│   │   ├── advanced_ensemble.py
│   │   └── final_solution.py
│   ├── submissions/
│   │   ├── simple_submission.py
│   │   └── quick_baseline.py
│   └── tracking/
│       ├── setup_experiment_tracking.py
│       ├── view_experiments.py
│       └── test_libraries.py
├── src/                          # Reusable modules (keep existing structure)
│   ├── analysis/
│   ├── diagnosis/
│   ├── tracking/
│   └── utils/
├── data/
│   ├── raw/                      # Original competition data
│   └── processed/                # Processed datasets (gitignored)
├── models/                       # Trained models and preprocessors (gitignored)
├── outputs/
│   ├── figures/                  # All PNG visualizations
│   └── submissions/              # Competition submission CSV files
├── docs/                         # All documentation
│   ├── README.md
│   ├── PROJECT_SUMMARY.md
│   └── analysis_reports/         # From answers/ directory
├── experiments/                  # MLflow, WandB tracking (keep existing)
│   ├── mlruns/
│   ├── wandb/
│   └── plots/
├── tests/                        # Test files (keep existing)
├── config/                       # Configuration files (keep existing)
├── .specify/                     # Specify framework (keep)
├── specs/                        # Feature specs (keep)
├── pyproject.toml                # Dependencies (root)
├── .gitignore                    # Updated with model/data patterns
└── .python-version               # Python version (root)
```

**Structure Decision**: Data analysis project with single root and organized subdirectories. Uses scripts/ for executable Python files and notebooks/ for Jupyter notebooks, following constitutional notebook-first principle. Separates source code (src/), outputs (outputs/), documentation (docs/), and experiments (experiments/).

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from data-model.md entities and quickstart.md validation steps
- Each entity type (Script, Notebook, Visualization, etc.) → categorization and move tasks
- Each directory in target structure → directory creation task
- Each validation step in quickstart.md → verification task
- .gitignore update task for large binary files

**Ordering Strategy**:
1. **Setup Phase**: Create all target directories first
2. **Categorization Phase**: Analyze and categorize files by purpose
3. **Move Phase**: Execute git mv operations to preserve history
   - Independent file moves marked [P] for parallel execution
   - Dependent moves (e.g., update imports) are sequential
4. **Update Phase**: Update .gitignore, create READMEs, generate DIRECTORY_STRUCTURE.md
5. **Validation Phase**: Run quickstart.md validation steps

**Task Categories**:
```
Setup (3-5 tasks):
- Create directory structure
- Backup current state
- Prepare .gitignore

Categorization (1 task):
- Analyze all files and assign purposes

Move Operations (10-15 tasks, many [P]):
- Move Python scripts to scripts/{purpose}/
- Move/rename notebooks to notebooks/ with numbering
- Move PNG files to outputs/figures/
- Move CSV files to outputs/submissions/
- Move .pkl files to models/
- Move documentation to docs/

Documentation (3-4 tasks):
- Create directory READMEs
- Generate DIRECTORY_STRUCTURE.md
- Update main README (if needed)

Validation (1 task):
- Execute quickstart.md validation checklist
```

**Estimated Output**: 18-25 numbered, ordered tasks in tasks.md

**Parallelization Opportunities**:
- File moves within same category can run in parallel (e.g., all script moves)
- Documentation tasks can run parallel to validation preparation
- README creation for different directories can be parallel

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) - research.md generated
- [x] Phase 1: Design complete (/plan command) - data-model.md and quickstart.md generated
- [x] Phase 2: Task planning complete (/plan command - approach described below)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS - No violations, supports all principles
- [x] Post-Design Constitution Check: PASS - Structure aligns with constitutional workflow
- [x] All NEEDS CLARIFICATION resolved - Clarifications completed in spec.md
- [x] Complexity deviations documented - None (straightforward file reorganization)

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*
