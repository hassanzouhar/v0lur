# 📋 Gold Standard Annotation Guidelines
**v0lur Milestone 7 Evaluation Framework**  
**Version:** 1.0  
**Date:** 2025-09-22

---

## 🎯 **Purpose**

These guidelines ensure consistent, high-quality manual annotations for evaluating v0lur's accuracy in:
- **Speaker attribution** (quote detection & span tagging)
- **Entity recognition** (NER accuracy & alias resolution)  
- **Stance classification** (stance detection & assignment)

---

## 🏷️ **Annotation Schema Overview**

Each message requires three types of annotations:

### **1. Span Annotations** (Speaker Attribution)
Tag text spans by who is speaking:
- `author`: Content authored by the message sender
- `quoted`: Direct quotes from someone else
- `forwarded`: Forwarded content from another source

### **2. Entity Annotations** (Named Entity Recognition)
Mark named entities with:
- `text`: Exact mention in the message
- `start`/`end`: Character positions
- `type`: PERSON, ORG, LOC, MISC
- `canonical_name`: Standardized form

### **3. Stance Annotations** (Stance Classification) 
Identify stance relationships:
- `speaker`: Who expresses the stance
- `target`: Entity being evaluated
- `label`: support, oppose, neutral
- `confidence`: 0.0-1.0 certainty level

---

## 📏 **Span Annotation Rules**

### **Author Spans**
- Default assumption: content is authored by message sender
- Include everything not explicitly quoted or forwarded
- Fill gaps between quotes/forwarded content

**Example:**
```
"I think Biden is doing well. As Obama said, 'Hope is important.' I agree completely."
```
- `author`: "I think Biden is doing well. As Obama said, "
- `quoted`: "Hope is important."  
- `author`: " I agree completely."

### **Quoted Spans**
Mark direct quotes with clear attribution:

**Clear Attribution:**
- `"Text"` with speaker identification
- `Person said: "text"`
- `According to X: "text"`

**Ambiguous Cases:**
- Scare quotes → `author` (unless clear speaker)
- Indirect quotes → `author`
- Paraphrasing → `author`

**Example:**
```
Trump tweeted: "This is fake news!" What a lie.
```
- `author`: "Trump tweeted: "
- `quoted`: "This is fake news!" (speaker: Donald Trump)
- `author`: " What a lie."

### **Forwarded Spans**
Mark content forwarded from other sources:
- `Forwarded from X:` prefix
- Clear forwarding metadata
- Reposted content with attribution

**Example:**
```
Forwarded from CNN: Breaking news about the election results...
```
- `forwarded`: "Breaking news about the election results..." (speaker: CNN)

---

## 🏢 **Entity Annotation Rules**

### **Entity Types**
- **PERSON**: Individuals (politicians, celebrities, etc.)
- **ORG**: Organizations, companies, political parties
- **LOC**: Locations, countries, regions  
- **MISC**: Everything else (events, products, etc.)

### **Boundary Rules**
- Include full entity mention: `"President Biden"` not `"Biden"`
- Exclude articles/determiners: `"the White House"` → `"White House"`
- Include titles when part of name: `"Senator Warren"`

### **Canonical Names**
Standardize to most common/official form:
- `"Obama"` → `"Barack Obama"`
- `"GOP"` → `"Republican Party"`
- `"US"` → `"United States"`

### **Difficult Cases**

**Pronouns:** Do NOT annotate unless unambiguous
```
"Biden spoke. He was confident." 
→ Only annotate "Biden", not "He"
```

**Nicknames/Slurs:** Annotate but use canonical name
```
"Sleepy Joe is wrong" 
→ Entity: "Sleepy Joe", Canonical: "Joe Biden"
```

**Partial Names:** Annotate if unambiguous in context
```
"Obama and Biden worked together"
→ Both are clear, annotate separately
```

---

## 🎯 **Stance Annotation Rules**

### **Stance Labels**
- **support**: Positive evaluation/endorsement
- **oppose**: Negative evaluation/criticism
- **neutral**: Factual reference without clear evaluation

### **Speaker Attribution**
- **author**: Default for most stances
- **quoted speaker**: When quote expresses stance
- **forwarded source**: When forwarded content expresses stance

### **Confidence Levels**
- **1.0**: Completely explicit ("I support X")
- **0.8-0.9**: Very clear but implicit  
- **0.6-0.7**: Somewhat clear with context
- **0.4-0.5**: Ambiguous/uncertain
- **< 0.4**: Too unclear to annotate (skip)

### **Evidence Spans**
Point to text supporting the stance judgment:
```
"Biden's policies are destroying America!"
→ Stance: oppose Biden
→ Evidence: full sentence (0-38)
```

### **Difficult Cases**

**Sarcasm:** Judge intended meaning, not literal
```
"Great job, Biden! Another brilliant move." (sarcastic)
→ Stance: oppose Biden
```

**Conditional Statements:** Usually neutral unless clear position
```
"If Biden wins, the economy will crash"
→ Stance: oppose Biden (implies negative outcome)
```

**Factual Reports:** Neutral unless editorial framing
```
"Biden signed the bill" → neutral
"Biden finally signed the bill" → slight oppose (implies delay)
```

---

## ⚠️ **Edge Cases & Common Errors**

### **Attribution Errors**
❌ **Wrong:** Attributing quoted stance to message author
```
"As Trump said, 'Biden is weak.' True!"
→ DON'T attribute "Biden is weak" to message author
→ DO attribute "True!" to message author
```

✅ **Right:** Separate quoted and author stances
```
Spans:
- author: "As Trump said, "
- quoted: "Biden is weak" (speaker: Donald Trump) 
- author: " True!"

Stances:
- Trump → Biden: oppose (from quoted span)
- Author → Trump: support (agrees with Trump)
```

### **Entity Confusion**
❌ **Wrong:** Multiple entities as single entity
```
"Obama and Biden" → single entity "Obama and Biden"
```

✅ **Right:** Separate entities
```
"Obama" + "Biden" → two separate PERSON entities
```

### **Stance Ambiguity**
When genuinely ambiguous, prefer higher confidence for:
- **Explicit statements** over implications
- **Clear context** over general references
- **Direct evaluation** over factual reporting

---

## 📊 **Quality Control**

### **Inter-Annotator Agreement**
- Each message annotated by 2+ people
- Disagreements resolved by discussion
- Target agreement: Cohen's κ > 0.8

### **Common Validation Checks**
1. **Span Coverage**: All text covered by exactly one span
2. **Entity Boundaries**: No overlapping entities
3. **Stance Consistency**: Stance targets match annotated entities
4. **Reference Resolution**: Canonical names are standardized

### **Annotation Review Process**
1. Initial annotation by trained annotator
2. Quality check by second annotator  
3. Disagreement resolution meeting
4. Final consensus annotation
5. Metadata and notes addition

---

## 🔧 **Annotation Tools**

### **Recommended Workflow**
1. Read entire message for context
2. Identify and mark spans (author/quoted/forwarded)
3. Mark entities with boundaries and types
4. Identify stance relationships with evidence
5. Add metadata and confidence scores
6. Review and validate annotation

### **Common Tools**
- **Text editor** with character counting
- **JSON formatter** for validation
- **Annotation tracking spreadsheet**

---

## 📝 **Example Annotations**

### **Simple Case**
```
Message: "Trump is the best president ever!"

Spans:
- author: "Trump is the best president ever!" (0-34)

Entities: 
- "Trump" (0-5) → PERSON → "Donald Trump"

Stances:
- author → Donald Trump: support (1.0)
```

### **Complex Case** 
```
Message: "As Biden said, 'We will rebuild better.' What nonsense from the Democrats!"

Spans:
- author: "As Biden said, " (0-15)
- quoted: "We will rebuild better." (16-38) [speaker: Joe Biden]
- author: " What nonsense from the Democrats!" (39-72)

Entities:
- "Biden" (3-8) → PERSON → "Joe Biden"  
- "Democrats" (57-66) → ORG → "Democratic Party"

Stances:
- author → Joe Biden: oppose (0.8) [evidence: "What nonsense"]
- author → Democratic Party: oppose (0.9) [evidence: "What nonsense from the Democrats"]
- Joe Biden → [rebuilding]: support (0.9) [from quoted span]
```

---

## 🎯 **Success Criteria**

High-quality annotations should achieve:
- **Span Attribution Accuracy > 95%**: Correct speaker identification
- **Entity Boundary Accuracy > 90%**: Correct entity boundaries
- **Stance Label Accuracy > 85%**: Correct support/oppose/neutral labels
- **Inter-Annotator Agreement > 0.8**: Consistent annotations across annotators

**Target: 200-300 annotated messages for robust evaluation**