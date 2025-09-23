# 📊 MILESTONE 7: EVALUATION FRAMEWORK IMPLEMENTATION PLAN
**Date:** 2025-09-22T16:56:00Z  
**Status:** Design Phase  
**Goal:** Create comprehensive validation & calibration system per spec requirements

---

## 🎯 **MILESTONE 7 REQUIREMENTS (From spec.md)**

From spec Section 12, Milestone 7 requirements:
- **Gold Set**: 200-300 messages with span-level annotations  
- **Attribution accuracy**: speaker attribution correctness
- **Entity accuracy**: NER precision/recall
- **Stance evaluation**: precision/recall with error analysis
- **Error analysis**: sarcasm misfires, nickname handling, quote misattribution
- **Quality measurement**: systematic accuracy assessment

---

## 🏗️ **EVALUATION FRAMEWORK ARCHITECTURE**

### **Component 1: Gold Standard Dataset**
```
data/gold_standard/
├── annotated_messages.json          # 200-300 manually annotated messages
├── annotation_guidelines.md         # Human annotation instructions
├── inter_annotator_agreement.json   # Quality control metrics
└── validation_splits/
    ├── dev_set.json                 # 50 messages for development
    ├── test_set.json                # 150-250 messages for final evaluation
    └── calibration_set.json         # Separate set for threshold tuning
```

### **Component 2: Evaluation Scripts**
```
scripts/evaluation/
├── evaluate.py                      # Main evaluation runner
├── metrics/
│   ├── attribution_metrics.py       # Speaker attribution accuracy
│   ├── entity_metrics.py            # NER precision/recall/F1
│   ├── stance_metrics.py            # Stance classification evaluation
│   └── error_analysis.py            # Detailed error categorization
├── calibration/
│   ├── threshold_tuner.py           # Automatic threshold optimization
│   └── confidence_calibration.py    # Score calibration analysis
└── reports/
    ├── generate_report.py           # Comprehensive evaluation report
    └── templates/
        └── evaluation_report.html    # HTML report template
```

### **Component 3: Annotation Schema**
```json
{
  "message_id": "unique_identifier",
  "text": "original message text",
  "ground_truth": {
    "spans": [
      {
        "start": 0,
        "end": 50,
        "text": "span text",
        "span_type": "author|quoted|forwarded",
        "speaker": "author|person_name|unknown",
        "confidence": 1.0
      }
    ],
    "entities": [
      {
        "text": "entity mention",
        "start": 10,
        "end": 20,
        "type": "PERSON|ORG|LOC|MISC",
        "canonical_name": "normalized_form"
      }
    ],
    "stance_edges": [
      {
        "speaker": "author|quoted_speaker",
        "target": "entity_name",
        "label": "support|oppose|neutral",
        "confidence": 0.8,
        "span_evidence": [0, 50]
      }
    ]
  },
  "metadata": {
    "annotator_id": "human_annotator_1",
    "annotation_time": "2025-09-22T16:56:00Z",
    "difficulty": "easy|medium|hard",
    "error_types": ["sarcasm", "nickname", "quote_attribution"]
  }
}
```

---

## 📐 **EVALUATION METRICS**

### **1. Attribution Accuracy**
```python
# Span-level attribution accuracy
def attribution_accuracy(predicted_spans, gold_spans):
    """
    Measures correct speaker attribution at span level.
    
    Returns:
        - span_accuracy: % of spans with correct speaker attribution
        - message_accuracy: % of messages with all spans correct
        - speaker_precision/recall: by speaker type (author/quoted/forwarded)
    """
```

**Key Metrics:**
- **Span Attribution Accuracy**: % spans with correct speaker
- **Quote Detection F1**: Precision/recall for detecting quoted content
- **Multi-speaker Accuracy**: % multi-speaker messages correctly handled
- **Author vs Non-author Classification**: Binary classification performance

### **2. Entity Accuracy** 
```python
# Standard NER evaluation
def entity_metrics(predicted_entities, gold_entities):
    """
    Standard NER evaluation with exact and partial matching.
    
    Returns:
        - exact_match_f1: Exact boundary + type match
        - partial_match_f1: Overlapping boundary + type match
        - type_accuracy: Correct type given correct boundary
        - boundary_accuracy: Correct boundary given correct type
    """
```

**Key Metrics:**
- **Exact Match F1**: Strict boundary + type match
- **Partial Match F1**: Overlapping spans with correct type
- **Type-specific F1**: Performance by entity type (PERSON, ORG, etc.)
- **Alias Resolution Accuracy**: Canonical name mapping correctness

### **3. Stance Evaluation**
```python
# Multi-level stance evaluation
def stance_metrics(predicted_edges, gold_edges):
    """
    Evaluate stance classification with speaker awareness.
    
    Returns:
        - overall_f1: Macro-averaged F1 across all stance labels
        - speaker_aware_f1: F1 considering correct speaker attribution
        - target_entity_f1: F1 for correct entity targeting
        - method_analysis: Performance by classification method (rules/MNLI/hybrid)
    """
```

**Key Metrics:**
- **Overall Stance F1**: Macro-averaged across support/oppose/neutral
- **Speaker-aware Stance F1**: Stance accuracy given correct attribution
- **High-confidence Accuracy**: Performance on confident predictions
- **Method Comparison**: Rules vs MNLI vs Hybrid performance

### **4. Error Analysis Categories**
```python
# Systematic error categorization
ERROR_CATEGORIES = {
    "sarcasm_misfires": "Sarcastic content misclassified as literal stance",
    "nickname_failures": "Failed to resolve nicknames/aliases to canonical forms",
    "quote_misattribution": "Incorrect speaker attribution for quotes",
    "boundary_errors": "Incorrect span boundaries",
    "type_confusion": "Wrong entity type classification",
    "stance_ambiguity": "Inherently ambiguous stance cases",
    "context_dependency": "Cases requiring broader context"
}
```

---

## 🔧 **IMPLEMENTATION PHASES**

### **Phase 1: Gold Standard Creation (Week 1-2)**

#### **Task 1.1: Data Selection Strategy**
```python
# Diverse sampling strategy
def select_annotation_candidates(df, target_count=250):
    """
    Select diverse, representative messages for annotation.
    
    Strategy:
        - 20% high-toxicity messages (edge cases)
        - 20% multi-entity messages (complexity)
        - 20% messages with quotes (attribution focus)
        - 20% different stance types (balance)
        - 20% random sample (baseline)
    """
```

#### **Task 1.2: Annotation Guidelines**
Create comprehensive guidelines covering:
- **Span Annotation Rules**: When to mark author vs quoted vs forwarded
- **Entity Annotation Standards**: Canonical name mapping, type decisions
- **Stance Annotation Criteria**: Support/oppose/neutral decision boundaries
- **Edge Case Handling**: Sarcasm, implied stance, ambiguous attribution

#### **Task 1.3: Annotation Tool**
Simple annotation interface (could be web-based or CLI):
```python
# CLI annotation tool
def annotate_message(message_text, message_id):
    """
    Interactive CLI tool for annotating messages.
    
    Features:
        - Span highlighting and labeling
        - Entity marking with type selection
        - Stance edge creation with confidence
        - Previous annotation review/edit
        - Progress tracking
    """
```

### **Phase 2: Evaluation Infrastructure (Week 2-3)**

#### **Task 2.1: Core Evaluation Engine**
```python
# scripts/evaluation/evaluate.py
class EvaluationEngine:
    """Main evaluation orchestrator."""
    
    def __init__(self, gold_standard_path, config_path):
        self.gold_data = self.load_gold_standard(gold_standard_path)
        self.config = Config(config_path)
        
    def run_full_evaluation(self, model_predictions):
        """Run comprehensive evaluation suite."""
        return {
            'attribution_metrics': self.evaluate_attribution(model_predictions),
            'entity_metrics': self.evaluate_entities(model_predictions),
            'stance_metrics': self.evaluate_stance(model_predictions),
            'error_analysis': self.analyze_errors(model_predictions),
            'calibration_analysis': self.analyze_calibration(model_predictions)
        }
```

#### **Task 2.2: Metrics Implementation**
Implement all metrics following standard evaluation practices:
- **Precision/Recall/F1** with micro/macro averaging
- **Confusion matrices** for multi-class problems  
- **Statistical significance testing** (paired t-test, bootstrap)
- **Confidence interval estimation** for robust reporting

#### **Task 2.3: Error Analysis Framework**
```python
# scripts/evaluation/error_analysis.py
def categorize_errors(predicted, gold, error_categories):
    """
    Systematic error categorization and analysis.
    
    Returns:
        - Error frequency by category
        - Representative examples per error type
        - Error correlation analysis
        - Improvement recommendations
    """
```

### **Phase 3: Calibration & Optimization (Week 3-4)**

#### **Task 3.1: Threshold Optimization**
```python
# scripts/evaluation/calibration/threshold_tuner.py
def optimize_thresholds(validation_data, metric='f1'):
    """
    Grid search over confidence thresholds to optimize specified metric.
    
    Optimizes:
        - stance_threshold: Stance classification confidence
        - topic_threshold: Topic assignment confidence
        - toxicity_threshold: Toxicity detection threshold
        - ner_confidence: Entity extraction confidence
    """
```

#### **Task 3.2: Confidence Calibration**
```python
def calibration_analysis(predicted_probs, true_labels):
    """
    Analyze and improve model confidence calibration.
    
    Techniques:
        - Platt scaling for probability calibration
        - Reliability diagrams
        - Expected Calibration Error (ECE)
        - Temperature scaling for neural models
    """
```

### **Phase 4: Reporting & Integration (Week 4)**

#### **Task 4.1: Comprehensive Reporting**
```python
# scripts/evaluation/reports/generate_report.py
def generate_evaluation_report(evaluation_results):
    """
    Generate detailed HTML evaluation report.
    
    Sections:
        - Executive summary with key metrics
        - Detailed performance breakdown
        - Error analysis with examples
        - Calibration plots and statistics
        - Recommendations for improvement
    """
```

#### **Task 4.2: CI/CD Integration**
```bash
# .github/workflows/evaluation.yml
# Automated evaluation on new commits/PRs
- name: Run Evaluation Suite
  run: |
    python scripts/evaluation/evaluate.py \
      --gold-standard data/gold_standard/test_set.json \
      --config config/config.yaml \
      --output evaluation_results.json
    
    # Fail if performance drops below threshold
    python scripts/evaluation/check_performance_regression.py
```

---

## 📊 **EXPECTED OUTCOMES**

### **Deliverables**
1. **Gold Standard Dataset**: 200-300 manually annotated messages
2. **Evaluation Suite**: Comprehensive automated evaluation framework
3. **Performance Baseline**: Current system accuracy measurements
4. **Optimization Results**: Calibrated thresholds and improved accuracy
5. **Quality Report**: Detailed analysis of strengths/weaknesses

### **Success Metrics**
- **Attribution Accuracy > 85%**: High-quality speaker attribution
- **Entity F1 > 90%**: Robust named entity recognition  
- **Stance F1 > 75%**: Reliable stance classification
- **Quote Detection F1 > 80%**: Accurate quote identification
- **Low Error Rates**: < 5% sarcasm misfires, < 3% nickname failures

### **Quality Assurance**
- **Inter-annotator Agreement > 0.8**: Consistent gold standard quality
- **Statistical Significance**: All comparisons with proper significance testing
- **Error Analysis Completeness**: All major error types identified and quantified
- **Calibration Quality**: Well-calibrated confidence scores

---

## 🚀 **IMPLEMENTATION ROADMAP**

### **Week 1: Gold Standard Foundation**
- [ ] Design annotation schema
- [ ] Select 250 messages for annotation
- [ ] Create annotation guidelines
- [ ] Build simple annotation tool
- [ ] Begin human annotation process

### **Week 2: Evaluation Infrastructure**
- [ ] Implement core metrics (attribution, entity, stance)
- [ ] Build evaluation engine framework
- [ ] Create error analysis system
- [ ] Set up automated evaluation pipeline

### **Week 3: Advanced Analysis**
- [ ] Complete human annotations (with quality control)
- [ ] Implement threshold optimization
- [ ] Add confidence calibration analysis
- [ ] Run full baseline evaluation

### **Week 4: Integration & Reporting**
- [ ] Generate comprehensive evaluation report
- [ ] Integrate into CI/CD pipeline
- [ ] Document findings and recommendations
- [ ] Plan next iteration improvements

---

## 🎯 **INTEGRATION WITH EXISTING SYSTEM**

### **Configuration Updates**
```yaml
# config/config.yaml - Add evaluation section
evaluation:
  gold_standard_path: data/gold_standard/test_set.json
  metrics_output_path: out/evaluation_results.json
  enable_error_analysis: true
  statistical_significance_threshold: 0.05
  bootstrap_iterations: 1000
```

### **Command Line Interface**
```bash
# New evaluation commands
python telegram_analyzer.py --evaluate --gold-standard data/gold_standard/test_set.json
python scripts/evaluation/evaluate.py --config config/config.yaml --detailed
python scripts/evaluation/calibration/optimize_thresholds.py --metric f1
```

### **Output Integration**
- **Evaluation results** added to standard output directory
- **Performance metrics** logged alongside processing statistics
- **Error reports** generated automatically for failed cases

---

## 📈 **EXPECTED IMPACT**

This evaluation framework will:

1. **🎯 Achieve Spec Compliance**: Complete Milestone 7 requirements fully
2. **📊 Enable Quality Measurement**: Systematic accuracy assessment
3. **🔧 Support Continuous Improvement**: Automated performance monitoring
4. **📋 Provide Research Validation**: Reproducible, rigorous evaluation
5. **🚀 Facilitate Production Deployment**: Confidence in system quality

**The evaluation framework will transform v0lur from a functional prototype into a research-grade, production-ready system with measurable quality guarantees.**

---

## 🎉 **MILESTONE COMPLETION CRITERIA**

Milestone 7 will be considered **COMPLETE** when:

- ✅ **Gold standard dataset**: 200+ annotated messages with high inter-annotator agreement
- ✅ **Evaluation suite**: All required metrics implemented and tested
- ✅ **Baseline results**: Full system evaluation with documented performance
- ✅ **Calibration**: Optimized thresholds for production use
- ✅ **Error analysis**: Comprehensive identification and categorization of failure modes
- ✅ **Integration**: Evaluation framework integrated into main system

**Target completion: 4 weeks from start**  
**Current spec compliance: 75% → 90%+ after completion**