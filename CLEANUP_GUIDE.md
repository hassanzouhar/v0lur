# 🧹 DEEP CLEANUP GUIDE - v0lur Storage Optimization
**Last Updated:** 2025-09-23T23:11:20Z  
**Target:** Storage optimization, performance improvement, development environment cleanup  
**Estimated Storage Savings:** 200MB - 2GB+ depending on output data

---

## 🎯 **CLEANUP STRATEGY OVERVIEW**

### **📊 Current Project Structure Analysis**
Based on your environment and the memory-safe improvements, this guide provides systematic cleanup recommendations to:

- **🗑️ Remove unnecessary files** (logs, temp data, old outputs)
- **⚡ Improve performance** (cache cleanup, dependency optimization)
- **💾 Optimize storage** (compress archives, clean checkpoints)
- **🔧 Maintain development efficiency** (keep essential files, preserve configs)

### **🚨 Safety-First Approach**
- ✅ **Green**: Safe to delete immediately
- 🟡 **Yellow**: Review before deletion (may contain useful data)
- 🔴 **Red**: Keep or backup first (critical for operation)

---

## 🗂️ **DIRECTORY-BY-DIRECTORY CLEANUP**

### **1. Output Data Directory (`out/`) - 🟡 SELECTIVE CLEANUP**

```bash
# Current structure (example):
out/
├── test-safe-20250924-0042/          # Recent test run - 🔴 KEEP
│   ├── checkpoints/                  # Memory-safe checkpoints - 🟡 REVIEW
│   ├── *.csv                        # Analysis outputs - 🟡 REVIEW  
│   ├── *.json                       # Analysis outputs - 🟡 REVIEW
│   └── *.parquet                    # Dataset - 🟡 REVIEW
├── old_run_20240815/                 # Old run - 🟡 REVIEW FOR DELETION
└── failed_run_20240801/              # Failed run - ✅ SAFE TO DELETE
```

**🧹 Cleanup Actions:**
```bash
# Remove failed runs (identifiable by incomplete output files)
find out/ -name "pipeline_status.json" -exec grep -l "failed\|error" {} \; | \
xargs -I {} dirname {} | xargs rm -rf

# Remove old runs older than 30 days (adjust as needed)
find out/ -type d -mtime +30 -name "*202*" -exec rm -rf {} \;

# Compress old but important runs (>7 days old)
for dir in out/*/; do
  if [[ $(stat -f %m "$dir") -lt $(date -j -v-7d +%s) ]]; then
    tar -czf "${dir%/}.tar.gz" -C "$(dirname "$dir")" "$(basename "$dir")"
    rm -rf "$dir"
  fi
done
```

**💾 Expected Savings:** 100MB - 1GB+ depending on number of old runs

---

### **2. Virtual Environment (`.venv/`) - 🔴 CAREFUL CLEANUP**

```bash
# Check .venv size
du -h .venv/
# Current: ~200-300MB (normal for ML dependencies)

# Safe cleanup actions:
.venv/bin/python -m pip cache purge          # Clear pip cache (~50MB)
find .venv/ -name "*.pyc" -delete           # Remove Python bytecode
find .venv/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

**⚠️ DO NOT DELETE:**
- `.venv/lib/python3.11/site-packages/` (core dependencies)
- `.venv/bin/` (executables)
- `.venv/pyvenv.cfg` (environment config)

**💾 Expected Savings:** 50-100MB

---

### **3. Reports Directory (`.reports/`) - ✅ SAFE CLEANUP**

```bash
# Current contents:
.reports/
├── freeze.pre.txt          # Dependency snapshot - 🟡 ARCHIVE
├── sitepackages.pre.txt    # Package listing - ✅ DELETE
└── sitepackages.summary.txt # Package summary - ✅ DELETE

# Cleanup commands:
rm .reports/sitepackages.pre.txt .reports/sitepackages.summary.txt
# Keep freeze.pre.txt for dependency troubleshooting
```

**💾 Expected Savings:** 5-10MB

---

### **4. Hidden Files and Cache - ✅ MOSTLY SAFE CLEANUP**

```bash
# macOS system files - ✅ SAFE TO DELETE
find . -name ".DS_Store" -delete
find . -name "._*" -delete

# Python cache files - ✅ SAFE TO DELETE  
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Textual UI cache (if exists) - ✅ SAFE TO DELETE
rm -rf ~/.textual/

# VS Code settings (if not needed) - 🟡 REVIEW
# rm -rf .vscode/  # Only if you don't use VS Code

# Git internals cleanup - ✅ SAFE OPTIMIZATION
git gc --prune=now
git remote prune origin  # If you have remotes
```

**💾 Expected Savings:** 10-50MB

---

### **5. Temporary and Log Files - ✅ SAFE CLEANUP**

```bash
# System temporary files
find . -name "*.tmp" -delete
find . -name "*.temp" -delete
find . -name "*.log" -mtime +7 -delete  # Logs older than 7 days

# Pipeline temporary files (if any exist)
find . -name "*_temp_*" -delete
find . -name "*.cache" -delete

# Jupyter notebook checkpoints (if any)
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null
```

**💾 Expected Savings:** 5-20MB

---

## 🔧 **ADVANCED CLEANUP PROCEDURES**

### **Checkpoint Management Strategy** 🟡

The memory-safe pipeline creates checkpoints. Here's how to manage them efficiently:

```bash
# Keep only the latest 2 checkpoint sets per run
for run_dir in out/*/; do
  if [[ -d "$run_dir/checkpoints" ]]; then
    cd "$run_dir/checkpoints"
    # Keep latest pipeline_status.json and most recent checkpoints
    ls -t *.parquet | tail -n +3 | xargs rm -f  # Keep latest 2 checkpoint files
    cd - > /dev/null
  fi
done

# Compress old checkpoint directories (>14 days)
find out/ -name "checkpoints" -type d -mtime +14 | while read dir; do
  tar -czf "${dir}.tar.gz" -C "$(dirname "$dir")" "$(basename "$dir")"
  rm -rf "$dir"
done
```

**💾 Expected Savings:** 50-200MB per old run

---

### **Dependency Optimization** 🟡

```bash
# Identify unused packages (requires pip-autoremove)
.venv/bin/pip install pip-autoremove
.venv/bin/pip-autoremove --list

# Clean pip cache
.venv/bin/pip cache purge

# Reinstall minimal requirements (if needed)
# .venv/bin/pip freeze > current_requirements.txt
# .venv/bin/pip uninstall -r current_requirements.txt -y
# .venv/bin/pip install -r requirements.txt
```

**⚠️ Only do dependency optimization if you're confident in your requirements.txt**

---

### **Data Deduplication** 🟡

```bash
# Find duplicate files in output directories
find out/ -type f -name "*.csv" -exec md5 {} \; | sort | uniq -d -w 32

# Remove duplicate checkpoint files with same content
find out/ -name "*checkpoint.parquet" -type f -exec md5 {} \; | \
sort | uniq -d -w 32 | cut -c 34- | xargs rm

# Compress similar analysis outputs
for format in "csv" "json"; do
  find out/ -name "*.$format" -size +1M -exec gzip {} \;
done
```

**💾 Expected Savings:** 20-100MB depending on duplication

---

## 📋 **AUTOMATED CLEANUP SCRIPT**

Create an automated cleanup script for regular maintenance:

```bash
#!/bin/bash
# save as: cleanup.sh

echo "🧹 Starting v0lur cleanup..."

# Safe deletions
echo "Removing system files..."
find . -name ".DS_Store" -delete
find . -name "._*" -delete
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Clear caches
echo "Clearing caches..."
if [[ -d .venv ]]; then
  .venv/bin/python -m pip cache purge > /dev/null 2>&1
fi

# Clean old outputs (optional - uncomment to enable)
# echo "Removing old runs (>30 days)..."
# find out/ -type d -mtime +30 -name "*202*" -exec rm -rf {} \; 2>/dev/null

# Git cleanup
echo "Git cleanup..."
git gc --prune=now --quiet

echo "✅ Cleanup completed!"
echo "💾 Run 'du -h .' to see current size"

# Make executable: chmod +x cleanup.sh
# Run: ./cleanup.sh
```

---

## ⚠️ **FILES TO NEVER DELETE**

### **🔴 Critical Files - DO NOT DELETE:**

```
config/config.yaml           # Core configuration
requirements.txt              # Dependencies
Makefile                      # Build automation
*.py files                    # Source code
WARP.md                      # Documentation
SPEC_GAP_ANALYSIS.md         # Analysis reports
BACKLOG.md                   # Development roadmap
```

### **🟡 Review Before Deleting:**

```
out/*/                       # May contain important analysis results
.venv/                       # Can be recreated but time-consuming
README_UI.md                 # UI documentation
*.md files                   # Documentation (review content first)
```

### **✅ Safe to Delete:**

```
.DS_Store                    # macOS system files
*.pyc                        # Python bytecode
__pycache__/                 # Python cache directories
*.log (old)                  # Log files >7 days
*.tmp                        # Temporary files
.reports/sitepackages.*      # Package listings
```

---

## 📊 **STORAGE OPTIMIZATION SUMMARY**

### **Before Cleanup (Estimated):**
- Source code: ~50MB
- Virtual environment: ~300MB  
- Output data: 100MB - 2GB+
- Cache/temp files: 50-100MB
- **Total: 500MB - 2.5GB+**

### **After Cleanup (Target):**
- Source code: ~50MB (unchanged)
- Virtual environment: ~250MB (-50MB cache cleanup)
- Output data: 50MB - 500MB (-50-75% from compression/archival)  
- Cache/temp files: <10MB (-80-90% reduction)
- **Total: 350MB - 800MB (30-70% reduction)**

---

## 🔄 **MAINTENANCE SCHEDULE**

### **Weekly (Automated):**
```bash
# Add to crontab: crontab -e  
# 0 2 * * 0 cd /Users/haz/c0de/v0lur && ./cleanup.sh
```

### **Monthly (Manual Review):**
- Review `out/` directory for old runs
- Check `.venv` size and clean if >500MB
- Archive or delete analysis results >30 days old
- Review and update this cleanup guide

### **Before Major Development:**
- Full cleanup including dependency review
- Backup important analysis results
- Fresh virtual environment if needed

---

## 🎯 **PERFORMANCE BENEFITS**

### **Expected Improvements:**
- **🚀 Faster startup:** Reduced cache scanning overhead
- **💾 Lower memory usage:** Less disk I/O for temporary files
- **⚡ Quicker git operations:** Smaller repository size
- **🔍 Easier navigation:** Cleaner directory structure
- **💪 Better performance:** Optimized virtual environment

### **Monitoring Storage:**
```bash
# Quick size check
du -h . | tail -1

# Detailed breakdown  
du -h --max-depth=1 | sort -hr

# Track specific directories
watch -n 5 "du -h out/ .venv/ | head -10"
```

---

## 💡 **BEST PRACTICES GOING FORWARD**

1. **🔄 Regular Cleanup:** Run automated cleanup weekly
2. **📁 Organized Outputs:** Use descriptive run names with dates
3. **💾 Selective Retention:** Archive important results, delete test runs
4. **🔧 Monitor Dependencies:** Regularly review `.venv` size  
5. **📊 Checkpoint Management:** Configure automatic checkpoint cleanup
6. **🚨 Backup Critical Data:** Before major cleanup operations

**This cleanup guide helps maintain a lean, efficient development environment while preserving essential functionality and important analysis results.** 🎯