# Git & GitHub Workflow Guide - NFL Predictions Project

## 🏈 Project Overview
This guide outlines Git and GitHub best practices for the NFL Predictions analytics platform. Follow these workflows to maintain clean, organized development and collaboration.

---

## 🎯 Quick Reference

### **Current Status:**
- **Repository**: `nfl-predictions`
- **Owner**: `gmalbert`
- **Default Branch**: `main`
- **Feature Branch**: `feature/enhanced-temporal-analysis` ✅

---

## 🚀 Standard Development Workflow

### **1. Starting New Features**
```bash
# Switch to main and get latest changes
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/descriptive-name

# Examples:
git checkout -b feature/playoff-predictions
git checkout -b feature/injury-analysis
git checkout -b feature/weather-integration
```

### **2. Making Changes**
```bash
# Work on your feature...
# Test thoroughly (run streamlit, validate models)

# Stage all changes
git add .

# Commit with descriptive message
git commit -m "feat: add playoff prediction model

- Implement Bayesian probability updates for playoff scenarios
- Add postseason performance metrics
- Integrate divisional standings calculations
- Update dashboard with playoff probability displays"
```

### **3. Pushing to GitHub**
```bash
# First push (sets up tracking)
git push -u origin feature/descriptive-name

# Subsequent pushes (after tracking is set)
git push
```

### **4. Creating Pull Request**
1. Go to GitHub repository
2. Click "Compare & pull request" button
3. Add detailed PR description
4. Review changes
5. Merge when ready

### **5. Cleanup After Merge**
```bash
# Switch back to main
git checkout main

# Pull merged changes
git pull origin main

# Delete local feature branch
git branch -d feature/descriptive-name
```

---

## 🏷️ Branch Naming Conventions

### **Feature Development:**
- `feature/enhanced-temporal-analysis` ✅
- `feature/monte-carlo-optimization`
- `feature/weather-data-integration`
- `feature/injury-report-analysis`

### **Bug Fixes:**
- `bugfix/streamlit-duplicate-ids` ✅
- `bugfix/xgboost-data-types`
- `bugfix/keyerror-monte-carlo`
- `bugfix/prediction-accuracy`

### **Documentation:**
- `docs/update-readme`
- `docs/api-documentation`
- `docs/user-guide`
- `docs/deployment-instructions`

### **Hotfixes (Critical):**
- `hotfix/critical-prediction-error`
- `hotfix/data-pipeline-failure`

---

## 📝 Commit Message Standards

### **Format:**
```
type: brief description (50 chars max)

- Detailed bullet points explaining changes
- What was modified and why
- Impact on system functionality
- Any breaking changes or migration notes
```

### **Types:**
- **feat**: New features or enhancements
- **fix**: Bug fixes and error corrections
- **docs**: Documentation updates
- **refactor**: Code restructuring without functionality changes
- **test**: Adding or updating tests
- **chore**: Maintenance tasks, dependency updates
- **perf**: Performance improvements
- **style**: Code formatting, linting fixes

### **Examples:**
```bash
# Feature addition
git commit -m "feat: add head-to-head historical analysis

- Implement calc_head_to_head_record() function
- Track team vs team historical performance
- Add headToHeadHomeTeamWinPct feature to models
- Improve underdog prediction accuracy by 2.3%"

# Bug fix
git commit -m "fix: resolve Streamlit duplicate element IDs

- Add unique keys to Monte Carlo number_input elements
- Fix 'mc_iter_1', 'mc_subset_1', 'mc_seed_1' for first section
- Fix 'mc_iter_2', 'mc_subset_2', 'mc_seed_2' for second section
- Enables both Monte Carlo sections to work simultaneously"

# Documentation
git commit -m "docs: update README with enhanced features

- Document current season performance tracking
- Add prior season records explanation
- Include head-to-head analysis details
- Update feature list with 50+ total features"
```

---

## 🔄 Workflow Scenarios

### **Scenario 1: Adding New Model Features**
```bash
git checkout main
git pull origin main
git checkout -b feature/weather-model-integration

# Develop weather features...
# Test with nfl-gather-data.py
# Validate with predictions.py dashboard
# Update README and documentation

git add .
git commit -m "feat: integrate weather data into prediction models"
git push -u origin feature/weather-model-integration

# Create PR → Review → Merge → Cleanup
```

### **Scenario 2: Fixing Critical Bugs**
```bash
git checkout main
git checkout -b hotfix/prediction-accuracy-issue

# Fix the critical issue...
# Test thoroughly
# Verify fix resolves problem

git add .
git commit -m "hotfix: correct probability calibration error"
git push -u origin hotfix/prediction-accuracy-issue

# Immediate PR → Fast review → Merge
```

### **Scenario 3: Documentation Updates**
```bash
git checkout main
git checkout -b docs/api-documentation

# Update documentation...
# Add code examples
# Improve explanations

git add .
git commit -m "docs: add comprehensive API documentation"
git push -u origin docs/api-documentation
```

---

## 🐛 Bug Fix Strategy & Workflow

### **Core Principle: Always Create New Branches**
**Never commit bug fixes directly to main** - even for small fixes. Here's why branches are essential:

- **Isolation**: Keep fixes separate until tested
- **History**: Track what was fixed and when
- **Rollback**: Easy to revert if fix causes issues
- **Review**: Maintain code quality through PR process
- **Collaboration**: Ready for team members if project grows

### **Bug Fix Decision Matrix**

| Bug Type | Branch Name | Priority | Process |
|----------|-------------|----------|---------|
| **Critical Production Issue** | `hotfix/descriptive-name` | Immediate | Fast-track PR → Merge ASAP |
| **Standard Bug** | `bugfix/descriptive-name` | Normal | Standard PR process |
| **Minor UI Issue** | `bugfix/ui-improvement` | Low | Can batch with other fixes |
| **Performance Issue** | `perf/optimization-name` | Medium | Test thoroughly before merge |

### **Bug Fix Workflow Examples**

#### **Small Dashboard Bug:**
```bash
# Found filter error in Streamlit dashboard
git checkout main
git pull origin main
git checkout -b bugfix/dashboard-filter-error

# Fix the specific issue
# Test thoroughly - run streamlit, verify fix works
# Check edge cases and error scenarios

git add .
git commit -m "fix: resolve dashboard filter TypeError

- Handle null values in team selection filter
- Add defensive programming for empty datasets  
- Prevent crash when no historical data available
- Add error logging for debugging future issues"

git push -u origin bugfix/dashboard-filter-error
# Create PR → Self-review → Merge → Delete branch
```

#### **Critical Model Error (Hotfix):**
```bash
# Production predictions are failing - urgent fix needed
git checkout main
git pull origin main
git checkout -b hotfix/xgboost-model-crash

# Fix critical issue immediately
# Test extensively with real data
# Verify predictions work end-to-end

git add .
git commit -m "hotfix: resolve XGBoost model loading compatibility

- Fix scikit-learn version compatibility issue
- Update model serialization to handle new sklearn format
- Add version checking to prevent future conflicts
- Restore full prediction pipeline functionality"

git push -u origin hotfix/xgboost-model-crash
# Immediate PR → Fast review → Merge → Monitor closely
```

#### **Multiple Related UI Fixes:**
```bash
# Several Streamlit UI issues found during testing
git checkout main
git pull origin main  
git checkout -b bugfix/streamlit-ui-improvements

# Fix all related UI issues in single branch
# Test each fix individually and together
# Ensure no regressions in existing functionality

git add .
git commit -m "fix: resolve multiple Streamlit UI and UX issues

- Fix duplicate element IDs in Monte Carlo sections
- Correct dataframe display width settings
- Resolve checkbox state persistence problems
- Improve error handling for missing data files
- Add loading indicators for long-running operations
- Enhance mobile responsiveness for key components"

git push -u origin bugfix/streamlit-ui-improvements
```

#### **Data Pipeline Bug:**
```bash
# Error in feature calculation causing model issues
git checkout main
git pull origin main
git checkout -b bugfix/feature-calculation-error

# Debug and fix the data pipeline issue
# Test with historical data
# Validate model performance unchanged
# Run full model training cycle to verify

git add .
git commit -m "fix: correct head-to-head calculation logic

- Fix off-by-one error in historical game filtering
- Ensure current game excluded from H2H calculations
- Add validation checks for edge cases (new teams, etc.)
- Update unit tests to catch similar issues
- Verify model accuracy maintained with corrected features"

git push -u origin bugfix/feature-calculation-error
```

### **Bug Fix Best Practices**

#### **Before Fixing:**
- [ ] **Reproduce the bug** consistently
- [ ] **Understand root cause** - don't just patch symptoms
- [ ] **Check for similar issues** elsewhere in codebase
- [ ] **Plan the fix approach** before coding

#### **During Fix:**
- [ ] **Fix only the specific issue** - avoid scope creep
- [ ] **Add defensive programming** to prevent recurrence
- [ ] **Consider edge cases** and error scenarios
- [ ] **Update related documentation** if needed

#### **After Fix:**
- [ ] **Test the specific fix** thoroughly
- [ ] **Run full system test** (streamlit dashboard, model training)
- [ ] **Verify no regressions** in other functionality
- [ ] **Document the fix** in commit message and PR

### **Branch Naming for Bugs**

```bash
# Good bug fix branch names
bugfix/dashboard-crash-empty-data       ✅
bugfix/monte-carlo-keyerror            ✅
bugfix/prediction-accuracy-regression   ✅
hotfix/model-loading-failure           ✅
perf/slow-feature-calculation          ✅

# Poor branch names
fix                                    ❌
bug                                    ❌
update                                 ❌
temp-fix                              ❌
```

### **When to Use Hotfix vs Bugfix**

#### **Hotfix (Urgent):**
- Production system is down/broken
- Critical predictions are failing
- Data corruption or loss risk
- Security vulnerabilities
- **Process**: Immediate fix → Fast-track PR → Emergency merge

#### **Bugfix (Standard):**
- Dashboard UI issues
- Non-critical calculation errors  
- Performance problems
- Minor feature malfunctions
- **Process**: Standard workflow → Normal PR review → Regular merge

### **Bug Fix Testing Checklist**

#### **NFL Predictions Specific:**
- [ ] **Model Training**: Run `python nfl-gather-data.py` successfully
- [ ] **Dashboard Loading**: Launch `streamlit run predictions.py` without errors
- [ ] **Predictions Generated**: Verify model outputs reasonable predictions
- [ ] **Monte Carlo Sections**: Test all feature selection interfaces
- [ ] **Data Loading**: Confirm all CSV files load properly
- [ ] **Historical Analysis**: Check betting analysis calculations
- [ ] **Edge Cases**: Test with missing data, empty datasets, invalid inputs

#### **General Testing:**
- [ ] **Specific Bug**: Original issue is resolved
- [ ] **No Regressions**: Existing functionality still works
- [ ] **Error Handling**: Graceful failure modes implemented
- [ ] **Performance**: No significant slowdowns introduced
- [ ] **Cross-Platform**: Works on different environments (if applicable)

---

## ⚠️ Safe Workflow Practices - Avoiding Conflicts

### **The Problem with Uncommitted Changes:**
**Never run `git pull origin main` with uncommitted local changes!** This can cause conflicts or prevent the pull entirely.

### **Safe Workflow Sequence:**

#### **Option 1: Clean Workspace (RECOMMENDED)**
```bash
# 1. Ensure clean workspace FIRST
git status                    # Check for uncommitted changes

# 2. If changes exist, handle them:
git add .                     # Stage changes
git commit -m "work in progress"  # OR
git stash push -m "WIP: feature work"  # Temporarily save changes

# 3. NOW it's safe to pull
git checkout main
git pull origin main          # No conflicts possible - clean workspace

# 4. Create feature branch
git checkout -b feature/new-work

# 5. If you stashed, restore your work
git stash pop                 # Only if you used stash
```

#### **Option 2: Stash-First Approach**
```bash
# Save current work temporarily
git stash push -m "WIP: working on feature"

# Safe to switch and pull now
git checkout main
git pull origin main

# Create new branch
git checkout -b feature/new-feature

# Restore your work (if needed on new branch)
git stash pop
```

### **Always Check Status First:**
```bash
git status                    # See what's modified
git diff                      # See specific changes
git stash list               # See any stashed work
```

### **Handle Uncommitted Changes Safely:**
```bash
# Option A: Commit work in progress
git add .
git commit -m "WIP: partial implementation"

# Option B: Stash for later
git stash push -m "Feature X progress - switching to bug fix"

# Option C: Create temporary branch
git checkout -b temp/save-work
git add . && git commit -m "Save work before switching"
git checkout main
```

### **Safe Branch Switching Sequence:**
```bash
# Instead of direct switch, use safe sequence:
git stash                     # Save work
git checkout main             # Switch safely  
git pull origin main          # Update safely
git checkout -b new-branch    # Create new branch
git stash pop                 # Restore work if needed
```

---

## 🗑️ Removing Files from Git Tracking

### **Complete File Removal Process:**

Sometimes you need to remove files that are already tracked by Git (large files, sensitive data, etc.).

#### **Step 1: Remove from Git tracking**
```bash
git rm --cached filename.txt          # Remove single file
git rm --cached -r foldername/         # Remove entire folder
git rm --cached *.log                  # Remove files by pattern
```

#### **Step 2: Add to .gitignore (prevent re-tracking)**
```bash
# Edit .gitignore file and add:
filename.txt
foldername/
*.log
```

#### **Step 3: Commit the removal**
```bash
git add .gitignore                     # Stage .gitignore changes
git commit -m "chore: remove tracked files and update .gitignore

- Remove filename.txt from version control
- Add to .gitignore to prevent future tracking"
```

#### **Step 4: Push to GitHub (REQUIRED!)**
```bash
git push origin branch-name            # Push the removal to remote
```

### **Why You MUST Push:**

#### **Without `git push`:**
- ❌ File still exists in GitHub repository  
- ❌ Other developers still see the file
- ❌ File appears in repository browser on GitHub
- ❌ File counts toward repository size

#### **With `git push`:**
- ✅ File removed from both local AND remote Git tracking
- ✅ File disappears from GitHub repository
- ✅ Clean state for all developers
- ✅ File no longer appears in GitHub repository browser

### **NFL Project Example:**
```bash
# Remove large data file from tracking
git rm --cached data_files/large_dataset.csv

# Add to .gitignore
echo "data_files/large_dataset.csv" >> .gitignore

# Commit the changes
git add .gitignore
git commit -m "chore: remove large dataset from version control

- Remove data_files/large_dataset.csv (50MB file)
- Add to .gitignore to prevent accidental re-commit
- File remains locally but not tracked by Git"

# Push to GitHub (REQUIRED!)
git push origin feature/enhanced-temporal-analysis
```

### **What Happens at Each Step:**

#### **After `git rm --cached`:**
- ✅ File removed from Git index (staging area)
- ✅ File still exists on your computer
- ❌ File still in GitHub repository

#### **After `git commit`:**
- ✅ Removal recorded in local Git history
- ❌ File still in GitHub repository

#### **After `git push`:**
- ✅ Removal applied to GitHub repository
- ✅ File disappears from GitHub
- ✅ Clean for everyone

### **Important Notes:**
- **File still exists locally** - you can still access it on your computer
- **File removed from future** - new clones won't get the file
- **File still in history** - previous commits still contain the file
- **Complete erasure** requires `git filter-branch` (advanced, rarely needed)

---

## 🎨 Advanced Git Tips

### **Viewing Changes:**
```bash
git status                    # See modified files
git diff                      # See unstaged changes
git diff --staged            # See staged changes
git log --oneline            # View commit history
```

### **Undoing Changes:**
```bash
git restore filename         # Undo unstaged changes
git restore --staged filename # Unstage changes
git reset HEAD~1             # Undo last commit (keep changes)
git reset --hard HEAD~1      # Undo last commit (lose changes)
```

### **Branch Management:**
```bash
git branch                   # List local branches
git branch -r                # List remote branches
git branch -d branch-name    # Delete merged branch
git branch -D branch-name    # Force delete unmerged branch
```

### **Stash Management:**
```bash
git stash                    # Save current work temporarily
git stash push -m "message"  # Save with descriptive message
git stash list               # See all stashed work
git stash pop                # Restore most recent stash
git stash apply stash@{0}    # Apply specific stash
git stash drop stash@{0}     # Delete specific stash
git stash clear              # Delete all stashes
```

---

## 📊 Project-Specific Considerations

### **Large Files (Git LFS):**
- Files >50MB automatically use Git LFS
- `.csv.gz` files are tracked with LFS
- Ensure Git LFS is installed: `git lfs install`

### **Files to Ignore (.gitignore):**
```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# Data files (large)
*.csv.gz
data_files/nfl_history_*.csv.gz

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.log
```

### **Pre-commit Checklist:**
- [ ] Code runs without errors
- [ ] Dashboard loads successfully (test with `streamlit run predictions.py`)
- [ ] Models train without KeyErrors or ValueErrors
- [ ] Monte Carlo sections work independently
- [ ] README updated if new features added
- [ ] No sensitive data committed (API keys, credentials)

---

## 🚨 Troubleshooting Common Issues

### **Merge Conflicts:**
```bash
# When conflicts occur during merge/rebase
git status                   # See conflicted files
# Manually edit files to resolve conflicts
git add .                    # Stage resolved files
git commit                   # Complete merge
```

### **Accidental Commits to Main:**
```bash
# Move commits to feature branch
git branch feature/accidental-work
git reset --hard HEAD~N      # N = number of commits to move
git checkout feature/accidental-work
```

### **Large File Issues:**
```bash
# Remove large file from history
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch large-file.csv' \
--prune-empty --tag-name-filter cat -- --all
```

---

## 🌟 Best Practices Summary

### **✅ Always Do:**
- Create feature branches for all development
- Write descriptive commit messages
- Test thoroughly before committing
- Pull latest changes before creating new branches
- Use meaningful branch names
- Review your own PRs before merging

### **❌ Never Do:**
- Commit directly to main branch
- Push untested code
- Use generic commit messages ("fix", "update")
- Commit large files without Git LFS
- Force push to shared branches
- Ignore merge conflicts

### **🎯 Project Goals:**
- Maintain 14.1% ROI betting strategy
- Keep codebase clean and documented
- Enable easy collaboration and feature additions
- Preserve data pipeline integrity
- Support rapid model iteration and testing

---

## 📞 Quick Command Reference

```bash
# Daily workflow
git checkout main && git pull origin main
git checkout -b feature/new-work
# ... work ...
git add . && git commit -m "feat: description"
git push -u origin feature/new-work

# Maintenance
git branch -d merged-branch
git remote prune origin
git gc --aggressive --prune=now

# Emergency
git stash                    # Save work temporarily
git checkout main            # Quick switch
git stash pop               # Restore work
```

---

*Last updated: October 29, 2025*  
*NFL Predictions Project - Enhanced Temporal Analysis*