#!/usr/bin/env python3
"""
Repository cleanup script for pre-GitHub sync.

This script helps organize the repository by:
1. Archiving outdated documentation
2. Organizing test scripts
3. Creating proper directory structure

Usage:
    python scripts/cleanup_repo.py [--dry-run] [--archive-docs] [--organize-scripts]
"""

import argparse
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent

# Files to archive
DOCS_TO_ARCHIVE = [
    "CLEANUP_SUMMARY.md",
    "MIGRATION_GUIDE.md",
    "Morphometrics.md",
    "STREET_SVF_DISCUSSION.md",
    "RIODASPEDRAS_EXECUTION_PLAN.md",
    "RIODASPEDRAS_SETUP.md",
    "RIODASPEDRAS_SETUP_COMPLETE.md",
    "RIODASPEDRAS_STREET_SVF_CHANGES.md",
    "MORPHOLOGY_METRICS_PLAN.md",
    "MORPHOLOGY_METRICS_IMPLEMENTATION_PLAN.md",
    "URBAN_MORPHOLOGY_PLAN.md",
    "SVF_OPTIMIZATION_PROPOSAL.md",
    "IMPLEMENTATION_STATUS.md",
    "NEXT_STEPS.md",
]

# Files to remove (development artifacts)
FILES_TO_REMOVE = [
    "claude.md",
    "PR_DESCRIPTION.md",
    "PUSH_CHECKLIST.md",
]

# Files to move to docs/guides/
DOCS_TO_GUIDES = [
    "COPACABANA_ANALYSIS_GUIDE.md",
    "RUN_ANALYSES.md",
]

# Test scripts to organize
TEST_SCRIPTS = [
    "test_gpu_svf_debug.py",
    "test_gpu_svf_small.py",
    "test_gpu_svf_selected_segments.py",
    "validate_gpu_svf.py",
    "compare_svf_cpu_gpu.py",
    "profile_svf_performance.py",
]

# Shell scripts to organize
SHELL_SCRIPTS = [
    "run_vidigal_svf_gpu_spacing_sweep.sh",
    "test_gpu_svf_riodaspedras.sh",
]

# Prototype scripts (consider removing)
PROTOTYPE_SCRIPTS = [
    "compute_svf_parallel_prototype.py",
]


def create_directories(dry_run=False):
    """Create necessary directory structure."""
    dirs = [
        PROJECT_ROOT / "docs" / "archive",
        PROJECT_ROOT / "docs" / "guides",
        PROJECT_ROOT / "scripts" / "tests",
        PROJECT_ROOT / "scripts" / "shell",
    ]
    
    for dir_path in dirs:
        if not dir_path.exists():
            if dry_run:
                print(f"[DRY RUN] Would create: {dir_path}")
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Created: {dir_path}")


def archive_docs(dry_run=False):
    """Archive outdated documentation files."""
    archive_dir = PROJECT_ROOT / "docs" / "archive"
    moved = 0
    
    for doc_file in DOCS_TO_ARCHIVE:
        src = PROJECT_ROOT / doc_file
        if src.exists():
            dst = archive_dir / doc_file
            if dry_run:
                print(f"[DRY RUN] Would move: {src} -> {dst}")
            else:
                shutil.move(str(src), str(dst))
                print(f"Moved: {src} -> {dst}")
                moved += 1
        else:
            print(f"Not found (skipping): {src}")
    
    return moved


def remove_files(dry_run=False):
    """Remove development artifact files."""
    removed = 0
    
    for file_name in FILES_TO_REMOVE:
        file_path = PROJECT_ROOT / file_name
        if file_path.exists():
            if dry_run:
                print(f"[DRY RUN] Would remove: {file_path}")
            else:
                file_path.unlink()
                print(f"Removed: {file_path}")
                removed += 1
        else:
            print(f"Not found (skipping): {file_path}")
    
    return removed


def move_to_guides(dry_run=False):
    """Move documentation files to docs/guides/."""
    guides_dir = PROJECT_ROOT / "docs" / "guides"
    moved = 0
    
    for doc_file in DOCS_TO_GUIDES:
        src = PROJECT_ROOT / doc_file
        if src.exists():
            dst = guides_dir / doc_file
            if dry_run:
                print(f"[DRY RUN] Would move: {src} -> {dst}")
            else:
                shutil.move(str(src), str(dst))
                print(f"Moved: {src} -> {dst}")
                moved += 1
        else:
            print(f"Not found (skipping): {src}")
    
    return moved


def organize_test_scripts(dry_run=False):
    """Move test scripts to scripts/tests/."""
    tests_dir = PROJECT_ROOT / "scripts" / "tests"
    moved = 0
    
    for script in TEST_SCRIPTS:
        src = PROJECT_ROOT / "scripts" / script
        if src.exists():
            dst = tests_dir / script
            if dry_run:
                print(f"[DRY RUN] Would move: {src} -> {dst}")
            else:
                shutil.move(str(src), str(dst))
                print(f"Moved: {src} -> {dst}")
                moved += 1
        else:
            print(f"Not found (skipping): {src}")
    
    return moved


def organize_shell_scripts(dry_run=False):
    """Move shell scripts to scripts/shell/."""
    shell_dir = PROJECT_ROOT / "scripts" / "shell"
    moved = 0
    
    for script in SHELL_SCRIPTS:
        src = PROJECT_ROOT / "scripts" / script
        if src.exists():
            dst = shell_dir / script
            if dry_run:
                print(f"[DRY RUN] Would move: {src} -> {dst}")
            else:
                shutil.move(str(src), str(dst))
                print(f"Moved: {src} -> {dst}")
                moved += 1
        else:
            print(f"Not found (skipping): {src}")
    
    return moved


def add_deprecation_notice(dry_run=False):
    """Add deprecation notice to deprecated scripts."""
    deprecated_script = PROJECT_ROOT / "scripts" / "analyze_sky_exposure.py"
    
    if not deprecated_script.exists():
        print(f"Deprecated script not found: {deprecated_script}")
        return False
    
    deprecation_notice = '''"""
DEPRECATED: This script is deprecated. Use analyze_sky_exposure_streets.py instead.

This script uses a fixed 45° sky exposure plane envelope, which is less flexible
than the ruleset-based approach in analyze_sky_exposure_streets.py.

For new analyses, use:
    python scripts/analyze_sky_exposure_streets.py --stl <stl_file> --footprints <footprints> --ruleset rio

This script is kept for backward compatibility only.
"""
'''
    
    if dry_run:
        print(f"[DRY RUN] Would add deprecation notice to: {deprecated_script}")
        return True
    
    # Read current file
    with open(deprecated_script, 'r') as f:
        content = f.read()
    
    # Check if already has deprecation notice
    if 'DEPRECATED' in content:
        print(f"Deprecation notice already exists in: {deprecated_script}")
        return True
    
    # Add deprecation notice at the top
    with open(deprecated_script, 'w') as f:
        f.write(deprecation_notice + content)
    
    print(f"Added deprecation notice to: {deprecated_script}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Cleanup repository for GitHub sync')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--archive-docs', action='store_true', help='Archive outdated documentation')
    parser.add_argument('--organize-scripts', action='store_true', help='Organize test and shell scripts')
    parser.add_argument('--remove-files', action='store_true', help='Remove development artifact files')
    parser.add_argument('--move-guides', action='store_true', help='Move docs to docs/guides/')
    parser.add_argument('--add-deprecation', action='store_true', help='Add deprecation notice to deprecated scripts')
    parser.add_argument('--all', action='store_true', help='Run all cleanup operations')
    
    args = parser.parse_args()
    
    if not any([args.archive_docs, args.organize_scripts, args.remove_files, 
                args.move_guides, args.add_deprecation, args.all]):
        parser.print_help()
        sys.exit(1)
    
    print("=" * 60)
    print("Repository Cleanup Script")
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
    print()
    
    # Create directories
    create_directories(dry_run=args.dry_run)
    print()
    
    # Run operations
    if args.all or args.archive_docs:
        print("Archiving outdated documentation...")
        moved = archive_docs(dry_run=args.dry_run)
        print(f"Archived {moved} files\n")
    
    if args.all or args.remove_files:
        print("Removing development artifacts...")
        removed = remove_files(dry_run=args.dry_run)
        print(f"Removed {removed} files\n")
    
    if args.all or args.move_guides:
        print("Moving documentation to docs/guides/...")
        moved = move_to_guides(dry_run=args.dry_run)
        print(f"Moved {moved} files\n")
    
    if args.all or args.organize_scripts:
        print("Organizing test scripts...")
        moved = organize_test_scripts(dry_run=args.dry_run)
        print(f"Moved {moved} test scripts\n")
        
        print("Organizing shell scripts...")
        moved = organize_shell_scripts(dry_run=args.dry_run)
        print(f"Moved {moved} shell scripts\n")
    
    if args.all or args.add_deprecation:
        print("Adding deprecation notices...")
        add_deprecation_notice(dry_run=args.dry_run)
        print()
    
    print("=" * 60)
    print("Cleanup complete!")
    if args.dry_run:
        print("This was a dry run. Use without --dry-run to apply changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
