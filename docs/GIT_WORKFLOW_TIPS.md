## One-line Git Workflow Checklist (Safe Feature Branching)

**Before you start work:**

1. `git checkout main && git pull --ff-only`
   - Ensures your local main branch is up to date with the remote and only fast-forwards (no merges or rebases). This prevents working on an outdated base and avoids merge conflicts later.
2. `git checkout -b my/feature-branch`
   - Creates a new feature branch from the latest main. This isolates your changes and makes it easy to track, review, and merge work.
3. Edit code → `git add` → `git commit` → `git push origin my/feature-branch`
   - Make your changes, stage them, commit with a message, and push to the remote feature branch. This keeps your work separate and safe from overwriting.

**Why this pattern prevents overwriting changes:**
- By always starting from an up-to-date main branch, you avoid working on stale code and reduce the risk of merge conflicts.
- Feature branches isolate your work, so local edits never overwrite changes on main or other branches.
- Pushing to a feature branch means your work is backed up and ready for review or merging, without affecting main until you open a pull request.

**Summary:**
Following this checklist ensures you never get the “your changes would be overwritten” error, keeps your work organized, and makes collaboration safe and predictable.
# Git Workflow Tips & Best Practices

## Understanding Git Terminology

### Current vs Incoming Changes

When merging branches, Git uses these terms:

- **Current (HEAD)**: The branch you're currently on (checked out)
- **Incoming**: The branch you're merging FROM

**Example:**
```powershell
git checkout update/dataframe-height  # Current (where you are)
git merge main                        # Incoming (what you're bringing in)
```

During a conflict:
```
<<<<<<< HEAD (Current Change)
Your current branch's code
=======
Incoming branch's code
>>>>>>> main (Incoming Change)
```

### Local vs Remote Branches

- **`main`** = Your local copy (might be outdated)
- **`origin/main`** = GitHub's version (always current)
- **`origin`** = The remote repository on GitHub

## Common Git Commands Explained

### git pull origin main

**What it does:** Downloads and merges changes from GitHub's main branch

**It's actually TWO commands:**
1. `git fetch origin main` - Downloads changes without modifying files
2. `git merge origin/main` - Merges those changes into your current branch

**Example:**
```powershell
# Before:
GitHub main:     A---B---C---D  (has latest changes)
Your local main: A---B          (outdated)

# After git pull origin main:
GitHub main:     A---B---C---D
Your local main: A---B---C---D  (now synchronized!)
```

**Why you need it:** Your local repository doesn't automatically update when changes are pushed to GitHub. You must manually pull them down.

**⚠️ Important: Will it overwrite my changes?**

**NO!** `git pull origin main` protects your work:

1. **If you have NO local changes:**
   - ✅ Simply updates your files to match GitHub
   - No conflicts, no overwrites

2. **If you have UNCOMMITTED changes (edited but not committed):**
   - ⚠️ Git will WARN you and REFUSE to pull
   - Error: "Your local changes would be overwritten by merge. Please commit or stash them."
   - **Your changes are PROTECTED**

3. **If you have COMMITTED changes:**
   - Git tries to MERGE GitHub's changes WITH yours
   - Different lines changed: Both changes are kept ✅
   - Same lines changed: You get a conflict to resolve manually
   - **Nothing is lost**

**Safe workflow before pulling:**
```powershell
# Option A: Commit your work first
git add .
git commit -m "My local changes"
git pull origin main  # Safe - merges your commit with GitHub

# Option B: Stash temporarily (see Commit vs Stash section below)
git stash  # Saves changes temporarily
git pull origin main
git stash pop  # Brings your changes back
```

**Safer alternative (two-step process):**
```powershell
git fetch origin main  # Download changes, don't merge yet
git log origin/main    # Review what's different
git merge origin/main  # Merge when ready
```

### Commit vs Stash

**Commit (Permanent)** - Saves changes to Git history forever

**When to use:**
- ✅ Work is complete or at a good stopping point
- ✅ You want to keep this change forever
- ✅ Ready to share with others (push to GitHub)
- ✅ Making progress you want to track

**Example:**
```powershell
git add raceAnalysis.py
git commit -m "Add get_dataframe_height function"
# This is now part of your project history FOREVER
```

**Characteristics:**
- Shows up in `git log`
- Can push to GitHub
- Part of branch history
- Other people can see it

---

**Stash (Temporary)** - Temporarily saves work-in-progress

**When to use:**
- ✅ Work is incomplete/messy
- ✅ Need to quickly switch to another branch
- ✅ Want to pull updates but have uncommitted changes
- ✅ Experimenting and want to save "just in case"

**Example:**
```powershell
git stash  # Saves current changes, reverts files to clean state
# Do other work...
git stash pop  # Brings changes back
```

**Characteristics:**
- Does NOT show in `git log`
- Can't push to GitHub
- Not part of any branch
- Temporary - usually applied back within minutes/hours
- Only you can see it (local only)

**Common stash commands:**
```powershell
git stash           # Save current changes
git stash list      # See all stashes
git stash pop       # Apply most recent stash and remove it
git stash apply     # Apply stash but keep it in list
git stash drop      # Delete a stash
git stash clear     # Delete all stashes
```

**Real-world stash example:**
```powershell
# You're halfway through coding when you need to fix a bug
git stash  # Hide unfinished work
git checkout main
git checkout -b hotfix-bug
# Fix bug, commit, merge
git checkout feature-branch
git stash pop  # Resume unfinished work
```

**Rule of Thumb:**
- **Commit** = "I want to keep this"
- **Stash** = "I need to set this aside temporarily"

**Safer alternative (two-step process):**
```powershell
git fetch origin main  # Download changes, don't merge yet
git log origin/main    # Review what's different
git merge origin/main  # Merge when ready
```

### git merge -X ours vs -X theirs

Automatically resolves conflicts by choosing one side:

**`git merge -X ours <branch-name>`**
- Keeps YOUR current branch's changes when conflicts occur
- Non-conflicting changes from incoming branch are still merged
- No conflict markers appear - automatic resolution

**`git merge -X theirs <branch-name>`**
- Keeps THEIR incoming branch's changes when conflicts occur
- Your current branch's conflicting changes are discarded

**Example:**
```powershell
# You're on update/dataframe-height
Branch A (yours):     README.md has "Logo width: 350px"
Branch B (incoming):  README.md has "Logo width: 450px"

git merge -X ours main
→ Result: "Logo width: 350px" (keeps YOUR version)

git merge -X theirs main
→ Result: "Logo width: 450px" (keeps THEIR version)
```

**⚠️ Warning:** These commands automatically resolve conflicts without showing you conflict markers. Use only when you're confident about which version to keep.

## Why Merge Conflicts Happen

### Common Causes:

**1. Multiple Active Branches Modifying Same Files**
```
update/logo-readme      → modifies README.md (adds logo)
update/dataframe-height → modifies README.md (adds function docs)
→ Conflict when merging both to main
```

**2. Not Pulling Latest Main Before Creating Branch**
```
You create branch from old local main
Someone else merges changes to GitHub main
Your branch is now based on outdated code
→ Conflict when trying to merge back
```

**3. Same Lines Changed Differently**
```
Branch A: Line 50 = "Logo width: 350px"
Branch B: Line 50 = "Logo width: 450px"
→ Git can't decide which to keep
```

## Solutions to Common Conflict Issues

### Option 1: Merge Branches Sequentially (Recommended)

```powershell
# Merge first branch to main
git checkout main
git pull origin main          # Get latest from GitHub
git merge update/logo-readme
git push origin main

# Then update second branch from main before merging
git checkout update/dataframe-height
git merge main                # Conflicts appear HERE - resolve once
git add .
git commit -m "Merge main into dataframe-height"

# Now merge to main (should be clean)
git checkout main
git merge update/dataframe-height
git push origin main
```

### Option 2: Combine Changes in One Branch

```powershell
# Merge all changes into one branch
git checkout update/dataframe-height
git merge update/logo-readme
# Resolve conflicts here
git add .
git commit -m "Combine logo and dataframe changes"

# Then merge to main once
git checkout main
git merge update/dataframe-height
```

### Option 3: Always Keep Current Changes

```powershell
git checkout update/dataframe-height
git merge -X ours origin/main  # Keep your branch's changes
git push origin update/dataframe-height
```

### Option 4: Update Local Main Before Creating Branches

```powershell
# ALWAYS do this before creating a new branch
git checkout main
git pull origin main           # ← Critical step
git checkout -b new-feature-branch
```

## Best Practices to Avoid Conflicts

### 1. Keep Local Main Synchronized

**Before creating any new branch:**
```powershell
git checkout main
git pull origin main
git checkout -b feature-branch
```

### 2. One Feature = One Branch

- Don't modify the same files (especially README.md) in multiple branches simultaneously
- Keep branches focused on a single feature or fix

### 3. Merge to Main Frequently

- Don't let branches diverge too much
- Smaller, frequent merges = fewer conflicts
- Less time for main to change while you work

### 4. Delete Merged Branches (CRITICAL for Avoiding Conflicts)

**Why delete both local AND GitHub branches?**
- Prevents accidental reuse of outdated branches
- Stale branches diverge further from main over time
- The longer they exist, the more conflicts accumulate
- Keeps workspace clean and reduces confusion
- Forces you to always start from fresh main

**Complete deletion workflow after PR is merged:**

```powershell
# 1. Delete on GitHub
# After merging PR, click "Delete branch" button on GitHub

# 2. Update local main with merged changes
git checkout main
git pull origin main

# 3. Delete local branch
git branch -d update/logo-readme  # Safe delete (prevents data loss)
# OR if needed:
git branch -D update/logo-readme  # Force delete

# 4. Clean up remote tracking references
git fetch --prune  # Removes references to deleted GitHub branches

# 5. Verify deletion
git branch -a  # Should not show deleted branch
```

**When to keep branches:**
- ❌ DON'T delete if PR is still open/under review
- ❌ DON'T delete if you're planning more work on that feature
- ✅ DO delete if PR is merged and feature is complete
- ✅ DO delete if you won't work on it again

**Pro Tip - Automate pruning:**
Add to your `.gitconfig`:
```
[fetch]
    prune = true  # Automatically prune on every fetch
```

### 5. Pull Main Into Your Branch Regularly

```powershell
# While working on feature-branch
git checkout feature-branch
git merge origin/main  # Keep branch up to date with main
```

### 6. Check Branch Status Before Creating PRs

```powershell
git fetch origin main
git log origin/main..HEAD  # See commits you're ahead
git log HEAD..origin/main  # See commits you're behind
```

## Quick Reference Commands

### Starting Work
```powershell
git checkout main
git pull origin main
git checkout -b new-feature
```

### Saving Work
```powershell
git add .
git commit -m "Description of changes"
git push origin feature-branch
```

### Updating Branch from Main
```powershell
git checkout feature-branch
git merge origin/main
# Resolve any conflicts
git push origin feature-branch
```

### Checking Status
```powershell
git status                    # Current branch and changes
git branch                    # List local branches
git branch -r                 # List remote branches
git log --oneline -10         # Recent commits
git diff origin/main          # What's different from main
```

### Cleaning Up
```powershell
git branch -d branch-name     # Delete local branch (safe)
git branch -D branch-name     # Force delete (caution!)
git fetch --prune             # Remove deleted remote branches
```

## Workflow Example

**Complete workflow for adding a feature:**

```powershell
# 1. Start with latest main
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b add-new-feature

# 3. Make changes, commit frequently
git add file1.py file2.md
git commit -m "Add new feature"

# 4. Push to GitHub
git push origin add-new-feature

# 5. Before creating PR, update from main
git fetch origin main
git merge origin/main
# Resolve any conflicts

# 6. Push updated branch
git push origin add-new-feature

# 7. Create PR on GitHub, merge when approved

# 8. Clean up
git checkout main
git pull origin main
git branch -d add-new-feature
```

## Troubleshooting

### "Your branch is behind origin/main"
```powershell
git pull origin main
```

### "Merge conflicts in README.md"
```powershell
# Open file, look for:
<<<<<<< HEAD
Your changes
=======
Their changes
>>>>>>> branch-name

# Edit to keep what you want, remove markers
git add README.md
git commit -m "Resolve merge conflict"
```

### "Can't merge - conflicts"
```powershell
# Option A: Keep your version
git merge -X ours origin/main

# Option B: Keep their version
git merge -X theirs origin/main

# Option C: Resolve manually
git merge origin/main
# Edit files, then:
git add .
git commit -m "Resolve conflicts"
```

### "I committed to wrong branch"
```powershell
# Move last commit to different branch
git checkout correct-branch
git cherry-pick <commit-hash>
git checkout wrong-branch
git reset --hard HEAD~1
```

## Key Takeaways

1. **Always pull main before creating branches**
2. **Use `origin/main` to get GitHub's latest version**
3. **`-X ours` keeps your changes, `-X theirs` keeps incoming changes**
4. **Merge frequently to avoid large conflicts**
5. **One branch = one feature** to minimize conflicts
6. **Local `main` ≠ GitHub `main`** until you pull

---

*Remember: Git conflicts are normal! They just mean multiple people (or you in multiple branches) changed the same code. The solution is always: decide which version to keep, or combine both.*
