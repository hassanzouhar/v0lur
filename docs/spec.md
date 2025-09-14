# Telegram Stance & Language Analysis Pipeline — Updated Development Spec

## 0) Purpose

Build a **neutral, reproducible analytics pipeline** to analyze a public Telegram channel and answer:

* Who the author **supports** and **opposes** over time.
* What **topics and issues** are discussed, including emerging ones.
* The **context** and **linguistic framing** of messages, including quotes and forwarded content.
* Which **external links/domains** are amplified.
* Produce outputs with traceable evidence to avoid over-attribution or bias.

The pipeline must prioritize **attribution accuracy**, **transparency**, and **config-driven repeatability**.

---

## 1) Scope

### In-scope (Phase 1)

* Normalize Telegram data (JSON/CSV).
* Entity extraction and aliasing to canonical forms.
* Stance classification per entity, with context-aware attribution.
* Hybrid **topic classification**:

  * Stable ontology for time series reporting.
  * Dynamic discovery to capture emerging topics.
* Quote detection and proper speaker attribution.
* Sentiment, toxicity, and stylistic features.
* Link and domain extraction.
* Aggregation outputs and visual sidecars.
* GPU acceleration (MPS for Apple, CUDA for Nvidia).
* CLI + config file driven.

### Future extensions

* Sarcasm/irony detection.
* Multimedia analysis (OCR, audio).
* Cross-lingual stance with automatic routing.

---

## 2) Data Model

### Input

Columns:

* `msg_id` (unique)
* `chat_id` (sender or channel ID)
* `date` (ISO8601 string)
* `text` (normalized message body)
* `media_type`, `media_url` (optional)
* `forwarded_from` (optional, for attribution)

### Output (per message)

| Column            | Type   | Description                                       |
| ----------------- | ------ | ------------------------------------------------- |
| `entities`        | JSON   | Extracted entities with canonical mapping         |
| `stance`          | JSON   | `{speaker, target, label, score, evidence_spans}` |
| `topics`          | JSON   | Scored list of topics                             |
| `top_topic_label` | string | Highest scoring topic                             |
| `sentiment_label` | string | Positive/Negative/Neutral                         |
| `sentiment_score` | float  | Confidence                                        |
| `toxicity_score`  | float  | Confidence                                        |
| `links`           | JSON   | URLs found                                        |
| `domains`         | JSON   | Parsed domains                                    |
| `language`        | string | Detected language                                 |
| `style_features`  | JSON   | Exclamation count, all-caps ratio, hedges, etc.   |

---

## 3) Processing Stages

### 3.1 Load & Normalize

* Support JSON, JSONL, or CSV formats.
* Coerce Telegram’s `list-of-spans` format into a single string.
* Limit text length (default 8k characters).

### 3.2 Language Detection

* Default to `langdetect`.
* Skip with `--skip-langdetect` if data is monolingual.

### 3.3 NER

* Default: Hugging Face token-classification model (e.g., `dslim/bert-base-NER`).
* Optional: spaCy model (`en_core_web_sm`).
* Normalize to `{PERSON, ORG, LOC, MISC}`.
* Deduplicate mentions per message.

### 3.4 Entity Aliasing

* Load from `aliases.json`.
* Map surface forms and nicknames → canonical entities with optional IDs.

Example:

```json
{
  "Donald Trump": {"aliases": ["Trump", "President Trump"], "type": "PERSON", "id": "Q22686"}
}
```

---

## 4) Quote Handling & Attribution

### Problem

Messages can contain:

* Author’s own words.
* Quoted opponents/allies.
* Forwarded messages.
* Mixed attributions.

Naive analysis falsely attributes quoted speech to the author.

### Solution

Treat messages as **multi-speaker** with span tagging:

* **Quote detection**:

  * Typographic quotes (`"..."`, `“...”`).
  * Quoted block lines (`>` prefix).
  * `Forwarded from` metadata.
  * Signature markers (`— NAME`).

* **Span tagging**:

  * Each sentence labeled `author`, `quoted(speaker=unknown|known)`, or `forwarded`.

* **Default stance rule**:

  * Exclude non-author spans unless there’s explicit framing.

---

## 5) Contextual Stance Classification

### Hybrid approach:

1. **Dependency-based rules** (fast, interpretable):

   * Parse author spans with spaCy.
   * Identify verbs/adjectives signaling stance:

     * *support, praise, condemn, oppose, criticize, corrupt, great*.
   * Handle negations (*"not support"* → flipped polarity).
   * Weight by dependency distance.

2. **Zero-shot fallback**:

   * Use `facebook/bart-large-mnli` or smaller (e.g., `distilbart-mnli`).
   * Template:

     > "The author expresses {support|oppose|neutral} toward {ENTITY}."

3. **Combined scoring**:

   * If both methods agree → high confidence.
   * If they conflict → downgrade to `neutral` or `unclear`.

### Event graph

* Create per-message graph:

  ```
  Author → [stance edge] → Entity
  ```
* Each edge contains:

  * Label (`support`, `oppose`, etc.)
  * Score
  * Evidence spans for audit.

---

## 6) Topics — Hybrid Approach

### Why hybrid:

Fixed topic lists are stable but blind to emerging issues.

### Steps:

1. **Ontology-based classification**:

   * Curated list like:

     ```json
     [
       {"label": "immigration", "keywords": ["immigration","border"]},
       {"label": "economy", "keywords": ["jobs","inflation"]}
     ]
     ```
   * Used for consistent longitudinal charts.

2. **Unsupervised discovery**:

   * Split messages into sentences.
   * Cluster using BERTopic or sentence embeddings + HDBSCAN.
   * Extract key phrases per cluster.

3. **Mapping clusters to ontology**:

   * Zero-shot classify clusters against ontology.
   * Unmatched clusters → labeled as `NEW_TOPIC_X`.

4. **Output**:

   * For each message, store both:

     * `ontology_topics`
     * `discovery_topics`

---

## 7) Links & Domains

* Regex extract URLs.
* Parse with `urlparse`.
* Strip `www.` prefix.
* Store both raw links and domain counts.

---

## 8) Style Features

* Exclamation count (`!`).
* All-caps ratio.
* Hedge words (e.g., "maybe", "perhaps").
* Superlatives ("best", "greatest").
* Persist in `style_features` JSON.

---

## 9) Aggregation Outputs

| File                         | Purpose                                    |
| ---------------------------- | ------------------------------------------ |
| `*_daily_summary.csv`        | Posts per day, avg sentiment, max toxicity |
| `*_entity_stance_counts.csv` | Total support/oppose per entity            |
| `*_entity_stance_daily.csv`  | Stance timeline per entity                 |
| `*_topic_share_daily.csv`    | Topic distribution over time               |
| `*_domain_counts.csv`        | Amplified domains                          |
| `*_top_toxic_messages.csv`   | Most toxic messages with context           |

---

## 10) Config Example

```yaml
io:
  input_path: data/channel.json
  format: json
  text_col: message
  id_col: message_id
  date_col: date
  out_path: out/posts_enriched.parquet

models:
  ner: dslim/bert-base-NER
  sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest
  toxicity: unitary/toxic-bert
  stance: facebook/bart-large-mnli
  topic: facebook/bart-large-mnli

processing:
  batch_size: 32
  prefer_gpu: true
  quote_aware: true
  skip_langdetect: true
  max_entities_per_msg: 3
  stance_threshold: 0.6

resources:
  aliases_path: config/aliases.json
  topics_path: config/topics.json
```

---

## 11) Evaluation

### Gold set

* 200–300 messages with span-level annotations:

  * Speaker, target entity, stance.

### Metrics

* **Attribution accuracy**: correct speaker credited.
* **Entity accuracy**: correct target identified.
* **Stance precision/recall**:

  * Prioritize **precision** for `support`/`oppose`.
  * Accept lower recall; ambiguous cases should be `unclear`.

### Error analysis

* Sarcasm misfires.
* Nicknames and group references missing from alias map.
* Misattributed quotes.

---

## 12) Milestones

| Milestone | Deliverable                                |
| --------- | ------------------------------------------ |
| M0        | Loader + NER + aliasing                    |
| M1        | Sentiment + toxicity + style               |
| M2        | Quote detection + speaker spans            |
| M3        | Dependency-based stance                    |
| M4        | Zero-shot stance integration               |
| M5        | Topic hybrid system (ontology + discovery) |
| M6        | Aggregations + sidecars                    |
| M7        | Validation + calibration                   |
| M8        | Optional dashboards                        |

---

## 13) Risks & Mitigations

| Risk                         | Mitigation                                             |
| ---------------------------- | ------------------------------------------------------ |
| Over-attribution from quotes | Strict default: exclude quotes unless explicit framing |
| Model bias                   | Document versions and thresholds; retrain periodically |
| Entity explosion             | Cap entities per message; prioritize canonical ones    |
| Runtime blowups              | Text clipping, batching, smaller MNLI models           |

---

## 14) Deliverables

* `telegram_analyzer.py` script.
* `config.yaml`, `aliases.json`, `topics.json`.
* Parquet + sidecar CSVs.
* `README.md` documenting pipeline and ethics.
* Optional `report.py` for quick visualizations.

---

## 15) Guiding Principles

* **Neutrality by design**: default to `unclear` when ambiguous.
* **Transparency**: evidence spans stored with every stance edge.
* **Hybrid thinking**: fixed ontology for stability, unsupervised discovery for novelty.
* **Speaker-aware**: never mix author words with quoted or forwarded voices.
* **Iterative refinement**: top entities and clusters logged for alias/topic updates.

---


