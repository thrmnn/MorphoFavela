# Repository Cleanup - Quick Reference

## Quick Start

Run the cleanup script with dry-run first to see what will happen:

```bash
python scripts/cleanup_repo.py --all --dry-run
```

If everything looks good, run it for real:

```bash
python scripts/cleanup_repo.py --all
```

## Manual Steps (Not Automated)

### 1. Update requirements.txt
Add clear comments for GPU dependencies:
```python
# GPU acceleration (optional - for GPU-accelerated SVF computation)
# Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install PyTorch3D: pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# torch>=2.0.0
# pytorch3d>=0.7.0
# See GPU_SETUP.md for detailed installation instructions
```

### 2. Update .gitignore
Add if not present:
```
.cursor/  # Cursor IDE files
*.tmp
*.bak
*~
```

### 3. Review and Update README.md
- Verify all examples are current
- Check that all paths are correct
- Ensure GPU instructions are clear

### 4. Create docs/guides/getting_started.md
Basic quick start guide for new users.

## What Gets Moved/Removed

### Archived (docs/archive/)
- Planning documents (*_PLAN.md, *_PROPOSAL.md)
- Status documents (IMPLEMENTATION_STATUS.md, NEXT_STEPS.md)
- Historical docs (CLEANUP_SUMMARY.md, MIGRATION_GUIDE.md)
- Area-specific setup docs (RIODASPEDRAS_*.md)

### Removed
- `claude.md` - AI context file
- `PR_DESCRIPTION.md` - Should be in git history
- `PUSH_CHECKLIST.md` - Development checklist

### Moved to docs/guides/
- `COPACABANA_ANALYSIS_GUIDE.md`
- `RUN_ANALYSES.md`

### Organized Scripts
- Test scripts → `scripts/tests/`
- Shell scripts → `scripts/shell/`

## After Cleanup

1. **Test that scripts still work**
   - Some scripts may have hardcoded paths that need updating
   - Test key workflows

2. **Update any hardcoded paths**
   - Check scripts that reference moved files
   - Update import paths if needed

3. **Commit changes**
   ```bash
   git add .
   git commit -m "Repository cleanup: organize docs and scripts for GitHub sync"
   ```

## Files to Keep in Root

- `README.md` - Main documentation
- `LICENSE` - License file
- `requirements.txt` - Dependencies
- `.gitignore` - Git ignore rules
- `ROADMAP.md` - Project roadmap (keep, but update status)
- `PRE_GITHUB_SYNC_ANALYSIS.md` - This analysis (can remove after cleanup)
- `CLEANUP_QUICK_REFERENCE.md` - This file (can remove after cleanup)

## Documentation Structure After Cleanup

```
docs/
├── guides/              # User-facing guides
│   ├── street_svf_usage.md
│   ├── gpu_setup.md
│   ├── sky_exposure_methodology.md
│   ├── copacabana_analysis_guide.md
│   └── run_analyses.md
├── technical/           # Technical documentation
│   ├── gpu_svf_implementation.md
│   ├── svf_optimization.md
│   └── ...
└── archive/             # Historical/outdated docs
    ├── cleanup_summary.md
    ├── migration_guide.md
    └── ...
```

## Scripts Structure After Cleanup

```
scripts/
├── [production scripts]  # Main analysis scripts
├── tests/                # Test/debug scripts
│   ├── test_gpu_svf_*.py
│   ├── validate_gpu_svf.py
│   └── ...
└── shell/                # Shell scripts
    ├── run_vidigal_svf_gpu_spacing_sweep.sh
    └── ...
```

## Verification Checklist

After cleanup, verify:
- [ ] All production scripts still work
- [ ] Documentation links are updated
- [ ] No broken imports
- [ ] README.md examples are correct
- [ ] .gitignore is complete
- [ ] No sensitive data in tracked files
- [ ] All tests pass (if applicable)
