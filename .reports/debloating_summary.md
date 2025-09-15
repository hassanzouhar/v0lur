# Debloating Summary Report

## Achievement: 33% Size Reduction (500MB Saved)

**Before:** 1.5GB site-packages  
**After:** 1.0GB site-packages  
**Savings:** 500MB (33% reduction)

## Major Optimizations

### 1. Topic Modeling Stack Replacement (~150MB+ saved)
- **Removed:** BERTopic + UMAP + HDBSCAN stack
- **Replaced with:** sentence-transformers + scikit-learn KMeans  
- **Benefit:** Lighter clustering while maintaining spec compliance

### 2. Parquet Engine Optimization (~107MB saved)
- **Removed:** PyArrow (112MB)
- **Replaced with:** FastParquet (~5MB)
- **Benefit:** Much smaller footprint, same functionality

### 3. Development Tools Separation (~100MB+ saved)  
- **Moved to requirements-dev.in:** pytest, mypy, ruff, black, plotly, etc.
- **Benefit:** Production environment only has runtime dependencies

### 4. spaCy Model Optimization (Future: ~42MB savings)
- **Current:** en_core_web_md (54MB) 
- **Planned:** en_core_web_sm (~12MB)
- **Status:** Pending validation of dependency parsing quality

## Spec Compliance Maintained

✅ **All required models preserved:**
- `dslim/bert-base-NER` 
- `cardiffnlp/twitter-roberta-base-sentiment-latest`
- `unitary/toxic-bert`
- `facebook/bart-large-mnli` 
- PyTorch with MPS acceleration

✅ **Core functionality validated:**
- MPS GPU acceleration works
- Transformers pipeline functional
- Sentence transformers operational  
- FastParquet read/write confirmed

## Dependencies That Remain Large (But Necessary)

1. **PyTorch (350MB)** - Required by spec, includes MPS for M1 acceleration
2. **SymPy (78MB)** - PyTorch dependency, unused by our pipeline but can't remove
3. **Transformers (112MB)** - Required by spec for all NLP models
4. **SciPy (101MB)** - Required by scikit-learn and sentence-transformers

## Files Changed

- `requirements.in` - Debloated core dependencies
- `requirements-dev.in` - Development tools separated  
- `requirements.txt` - Recompiled with optimizations
- `requirements-minimal.in` - Test version created

## Validation Status

✅ Core imports successful  
✅ PyTorch MPS acceleration confirmed  
✅ Transformers models loadable  
✅ FastParquet operational  
✅ All spec-required functionality preserved

## Next Steps for Further Optimization

1. **Switch to en_core_web_sm** (saves ~42MB)  
2. **Remove unused dev packages from any production installs**
3. **Consider optional extras for heavy features** (HDBSCAN, PyArrow)
4. **Set up CI size gate** to prevent future bloat