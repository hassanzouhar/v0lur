# Critical Issues to Create on GitHub

**Generated:** 2025-11-01
**Branch:** claude/fix-critical-bugs-011CUgXbpxKvYZiFRz3AuFtL

## Issue 1: Bare Exception Handler in Discovery Topic Processor

**Severity:** 🔴 Critical
**File:** `src/raigem0n/processors/discovery_topic_processor.py:254`

### Description
The code uses a bare `except:` clause which catches all exceptions including `KeyboardInterrupt`, `SystemExit`, and other critical system exceptions that should not be caught.

### Current Code
```python
try:
    repr_docs = self.bertopic_model.get_representative_docs(topic_id)
    # ...
except:  # ⚠️ CRITICAL: Catches ALL exceptions
    topic_detail['representative_docs'] = []
```

### Fix
Change to `except Exception as e:` to avoid catching system exceptions.

### Impact
- Can suppress critical interrupts (Ctrl+C)
- Makes debugging difficult
- Violates Python best practices

---

## Issue 2: Division by Zero in Logging Statements

**Severity:** 🔴 Critical
**Files:** Multiple processors

### Description
Several processors perform division operations in logging statements without checking if the DataFrame is empty, which will raise `ZeroDivisionError`.

### Affected Files
- `src/raigem0n/processors/quote_processor.py:397-398`
- `src/raigem0n/processors/links_processor.py:271`
- `src/raigem0n/processors/style_processor.py:333`
- `src/raigem0n/processors/stance_processor.py:486`

### Example
```python
# quote_processor.py:397
logger.info(f"Messages with quotes: {messages_with_quotes}/{len(df)} ({messages_with_quotes/len(df)*100:.1f}%)")
# ⚠️ Will crash if df is empty
```

### Fix
Add checks before division:
```python
if len(df) > 0:
    logger.info(f"Messages with quotes: {messages_with_quotes}/{len(df)} ({messages_with_quotes/len(df)*100:.1f}%)")
```

### Impact
- Runtime crashes on empty datasets
- Poor user experience
- Data pipeline failures

---

## Issue 3: Incorrect DataFrame API Usage

**Severity:** 🔴 Critical
**File:** `src/raigem0n/processors/links_processor.py:390`

### Description
The code uses `DataFrame.get()` method which doesn't exist on pandas DataFrames - `.get()` is a dictionary method.

### Current Code
```python
df_with_links['date'] = pd.to_datetime(df_with_links.get('date', pd.Timestamp.now()))
```

### Fix
```python
if 'date' in df_with_links.columns:
    df_with_links['date'] = pd.to_datetime(df_with_links['date'])
else:
    df_with_links['date'] = pd.Timestamp.now()
```

### Impact
- Will raise `AttributeError` at runtime
- Breaks links processing pipeline
- No temporal analysis possible

---

## Issue 4: Unused Imports Across Processors

**Severity:** 🟡 High Priority
**Files:** Multiple processor files

### Description
Multiple processors import `AutoModelForSequenceClassification` and `AutoTokenizer` but never use them, as they use the higher-level `pipeline` API instead.

### Affected Files
- `src/raigem0n/processors/sentiment_processor.py:7`
- `src/raigem0n/processors/ner_processor.py:7`
- `src/raigem0n/processors/stance_processor.py:9`
- `src/raigem0n/processors/toxicity_processor.py:7`
- `src/raigem0n/processors/topic_processor.py:7` (also unused `import torch`)

### Fix
Remove unused imports from all affected files.

### Impact
- Code clutter and confusion
- Slower import times
- Misleading for new developers

---

## Issue 5: Inconsistent Exception Handling

**Severity:** 🟡 High Priority
**Scope:** Codebase-wide

### Description
Exception handling patterns vary across the codebase with no consistent standard for which exception types to catch or which logging level to use.

### Current Inconsistencies
- Some use `except Exception as e:` ✅
- Some use bare `except:` ❌
- Inconsistent use of `logger.warning()` vs `logger.error()`
- No standard pattern for exception severity

### Fix
Standardize on:
- Always use `except Exception as e:` (never bare `except:`)
- Use `logger.error()` for initialization failures
- Use `logger.warning()` for expected/recoverable errors
- Use `logger.exception()` when full traceback is needed

### Impact
- Makes debugging harder
- Inconsistent error reporting
- Can catch unintended exceptions

---

**All issues will be fixed in PR from branch:** `claude/fix-critical-bugs-011CUgXbpxKvYZiFRz3AuFtL`
