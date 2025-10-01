#!/usr/bin/env python3
"""
Remove SVC-related content from all three notebooks (07, 08, 09).
User request: "svc 영역은 제거" (Remove SVC area)
"""

import json
import sys

def remove_svc_from_notebook_07(notebook_path):
    """Remove SVC sections from notebook 07_hyperparameter_optimization.ipynb"""
    print(f"Processing {notebook_path}...")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells_to_keep = []
    skip_count = 0

    for cell in nb['cells']:
        # Check if this is an SVC-related cell
        is_svc_cell = False

        # Check markdown cells for "SVC" or "Support Vector" section headers
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            # Look for section headers about SVC
            if '## 6. SVC' in source or '6. SVC' in source or 'Support Vector Classifier' in source:
                is_svc_cell = True
                print(f"  Removing SVC section header cell")

        # Check code cells for SVC-specific optimization
        elif cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Look for SVC optimization code (objective_svc, study_svc, etc.)
            if ('objective_svc' in source or
                'study_svc' in source or
                "name='SVC'" in source or
                "model_name = 'svc'" in source):
                is_svc_cell = True
                print(f"  Removing SVC optimization code cell")
            # Also remove cells that only print/visualize SVC results
            elif "'svc'" in source.lower() and ('print' in source or 'plot' in source):
                # Check if this is ONLY about SVC (not a comparison table)
                if source.count('svc') > 2 and 'all_results' not in source:
                    is_svc_cell = True
                    print(f"  Removing SVC-specific output cell")

        if not is_svc_cell:
            # Keep the cell, but modify if it contains SVC references in lists/comparisons
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                # Fix model lists that include SVC
                if "model_names = ['qda', 'svc'," in source:
                    new_source = source.replace(
                        "model_names = ['qda', 'svc', 'randomforest', 'decisiontree']",
                        "model_names = ['qda', 'randomforest', 'decisiontree']"
                    )
                    if new_source != source:
                        cell['source'] = new_source.split('\n')
                        if cell['source'][-1] == '':
                            cell['source'] = cell['source'][:-1]
                        print(f"  Modified model list to remove 'svc'")

                # Fix introduction tables
                if 'Model' in source and 'QDA' in source and 'SVC' in source:
                    # This is likely the comparison table in introduction
                    # Remove SVC row from markdown table if present
                    lines = source.split('\n')
                    new_lines = [line for line in lines if 'SVC' not in line or 'Support Vector' not in line]
                    if len(new_lines) != len(lines):
                        cell['source'] = new_lines
                        print(f"  Removed SVC from comparison table")

            cells_to_keep.append(cell)
        else:
            skip_count += 1

    nb['cells'] = cells_to_keep

    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"  ✅ Removed {skip_count} SVC-related cells from notebook 07")
    return skip_count


def remove_svc_from_notebook_08(notebook_path):
    """Remove SVC references from notebook 08_feature_engineering.ipynb"""
    print(f"\nProcessing {notebook_path}...")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified_cells = 0

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Fix model_names list
            if "model_names = ['qda', 'svc', 'randomforest', 'decisiontree']" in source:
                new_source = source.replace(
                    "model_names = ['qda', 'svc', 'randomforest', 'decisiontree']",
                    "model_names = ['qda', 'randomforest', 'decisiontree']"
                )
                cell['source'] = new_source.split('\n')
                if cell['source'][-1] == '':
                    cell['source'] = cell['source'][:-1]
                print(f"  Modified model_names list to remove 'svc'")
                modified_cells += 1

            # Update counts in comments/prints (4 models → 3 models)
            if '4 models' in source or 'four models' in source.lower():
                new_source = source.replace('4 models', '3 models').replace('four models', 'three models')
                if new_source != source:
                    cell['source'] = new_source.split('\n')
                    if cell['source'][-1] == '':
                        cell['source'] = cell['source'][:-1]
                    print(f"  Updated model count from 4 to 3")
                    modified_cells += 1

    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"  ✅ Modified {modified_cells} cells in notebook 08")
    return modified_cells


def remove_svc_from_notebook_09(notebook_path):
    """Remove SVC references from notebook 09_final_ensemble.ipynb"""
    print(f"\nProcessing {notebook_path}...")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified_cells = 0

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Fix model loading list
            if "for name in ['qda', 'svc', 'randomforest', 'decisiontree']:" in source:
                new_source = source.replace(
                    "for name in ['qda', 'svc', 'randomforest', 'decisiontree']:",
                    "for name in ['qda', 'randomforest', 'decisiontree']:"
                )
                cell['source'] = new_source.split('\n')
                if cell['source'][-1] == '':
                    cell['source'] = cell['source'][:-1]
                print(f"  Modified model loading list to remove 'svc'")
                modified_cells += 1

            # Fix all_estimators list for stacking
            if "all_estimators = [" in source and "('svc', models['svc'])" in source:
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    if "('svc', models['svc'])" not in line:
                        new_lines.append(line)
                    else:
                        print(f"  Removed SVC from all_estimators list")

                new_source = '\n'.join(new_lines)
                cell['source'] = new_source.split('\n')
                if cell['source'][-1] == '':
                    cell['source'] = cell['source'][:-1]
                modified_cells += 1

            # Fix submission_models dictionary
            if "submission_models = {" in source and "'svc_optimized': models['svc']" in source:
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    if "'svc_optimized': models['svc']" not in line:
                        new_lines.append(line)
                    else:
                        print(f"  Removed SVC from submission_models")

                new_source = '\n'.join(new_lines)
                cell['source'] = new_source.split('\n')
                if cell['source'][-1] == '':
                    cell['source'] = cell['source'][:-1]
                modified_cells += 1

            # Update counts (4 models → 3 models, 7 submissions → 6 submissions)
            new_source = source
            if '4 models' in source or 'four models' in source.lower():
                new_source = new_source.replace('4 models', '3 models').replace('four models', 'three models')
            if '7 submission' in source or '7 total' in source:
                new_source = new_source.replace('7 submission', '6 submission').replace('7 total', '6 total')
                new_source = new_source.replace('4 individual + 3 ensemble = 7', '3 individual + 3 ensemble = 6')

            if new_source != source:
                cell['source'] = new_source.split('\n')
                if cell['source'][-1] == '':
                    cell['source'] = cell['source'][:-1]
                print(f"  Updated model/submission counts")
                modified_cells += 1

        # Fix markdown cells
        elif cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])

            # Update counts in markdown
            new_source = source
            if '4 models' in source or 'four models' in source.lower():
                new_source = new_source.replace('4 models', '3 models').replace('four models', 'three models')
            if '7 submission' in source or '7 files' in source:
                new_source = new_source.replace('7 submission', '6 submission').replace('7 files', '6 files')
            if 'ALL 4 models' in source:
                new_source = new_source.replace('ALL 4 models', 'ALL 3 models')
            if '4 individual + 3 ensemble = 7' in source:
                new_source = new_source.replace('4 individual + 3 ensemble = 7', '3 individual + 3 ensemble = 6')

            if new_source != source:
                cell['source'] = new_source.split('\n')
                if cell['source'][-1] == '':
                    cell['source'] = cell['source'][:-1]
                print(f"  Updated markdown counts")
                modified_cells += 1

    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"  ✅ Modified {modified_cells} cells in notebook 09")
    return modified_cells


def main():
    """Main execution"""
    notebooks = [
        '/home/e7217/projects/dacon-236590/notebooks/07_hyperparameter_optimization.ipynb',
        '/home/e7217/projects/dacon-236590/notebooks/08_feature_engineering.ipynb',
        '/home/e7217/projects/dacon-236590/notebooks/09_final_ensemble.ipynb'
    ]

    print("="*80)
    print("Removing SVC content from notebooks (User request: svc 영역은 제거)")
    print("="*80)

    total_changes = 0

    try:
        # Process each notebook
        total_changes += remove_svc_from_notebook_07(notebooks[0])
        total_changes += remove_svc_from_notebook_08(notebooks[1])
        total_changes += remove_svc_from_notebook_09(notebooks[2])

        print("\n" + "="*80)
        print(f"✅ SUCCESS: Made {total_changes} total changes across all 3 notebooks")
        print("="*80)
        print("\nChanges made:")
        print("  • Notebook 07: Removed entire SVC optimization section")
        print("  • Notebook 08: Removed 'svc' from model_names list")
        print("  • Notebook 09: Removed SVC from ensemble creation and submissions")
        print("  • Updated all model counts from 4 to 3")
        print("  • Updated submission counts from 7 to 6")
        print("\nResult: Only QDA, RandomForest, and DecisionTree remain")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
