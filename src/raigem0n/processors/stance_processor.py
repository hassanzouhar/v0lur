"""Stance classification processor using hybrid dependency rules + zero-shot MNLI."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class StanceProcessor:
    """Hybrid stance classification using dependency rules + zero-shot MNLI."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        stance_threshold: float = 0.6,
        dep_rules_weight: float = 0.6,
        mnli_weight: float = 0.4,
        device: Optional[str] = None,
    ) -> None:
        """Initialize stance processor.
        
        Args:
            model_name: Hugging Face model name for zero-shot classification
            stance_threshold: Minimum confidence threshold for stance classification
            dep_rules_weight: Weight for dependency rules in hybrid scoring
            mnli_weight: Weight for MNLI model in hybrid scoring
            device: Device to run model on (cuda, mps, cpu)
        """
        self.model_name = model_name
        self.stance_threshold = stance_threshold
        self.dep_rules_weight = dep_rules_weight
        self.mnli_weight = mnli_weight
        self.device = device
        
        # Initialize models
        self.nlp = None
        self.mnli_pipeline = None
        self._load_models()
        
        # Stance lexicons (from spec)
        self.support_verbs = {
            'support', 'back', 'endorse', 'praise', 'defend', 'champion',
            'approve', 'favor', 'promote', 'advocate', 'celebrate', 'love',
            'admire', 'respect', 'trust', 'appreciate', 'agree'
        }
        
        self.oppose_verbs = {
            'oppose', 'criticize', 'condemn', 'attack', 'denounce', 'reject',
            'hate', 'despise', 'disapprove', 'disagree', 'fight', 'resist',
            'protest', 'slam', 'blast', 'destroy', 'corrupt', 'crooked'
        }
        
        self.negative_words = {
            'not', 'never', 'no', "don't", "doesn't", "won't", "can't", 
            "shouldn't", "wouldn't", "isn't", "aren't", "wasn't", "weren't"
        }
        
        self.hedge_words = {
            'might', 'maybe', 'perhaps', 'possibly', 'seems', 'appears',
            'probably', 'likely', 'could', 'would', 'should'
        }

    def _load_models(self) -> None:
        """Load spaCy and MNLI models."""
        try:
            # Load spaCy for dependency parsing
            logger.info("Loading spaCy model for dependency parsing")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
            
            # Load MNLI model for zero-shot classification
            logger.info(f"Loading MNLI model: {self.model_name}")
            self.mnli_pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("MNLI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def classify_stance(
        self, 
        text: str, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Classify stance towards entities in text.
        
        Args:
            text: Message text to analyze
            entities: List of extracted entities from NER
            
        Returns:
            List of stance edges with speaker, target, label, score, evidence
        """
        if not self.nlp or not self.mnli_pipeline:
            raise RuntimeError("Models not loaded")
        
        stance_edges = []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # For each entity, determine stance
        for entity in entities:
            entity_text = entity['text']
            entity_type = entity.get('type', 'MISC')
            
            # Skip non-person/org entities for stance analysis
            if entity_type not in ['PERSON', 'ORG']:
                continue
            
            # Get stance using hybrid approach
            stance_result = self._hybrid_stance_classification(
                text, doc, entity_text
            )
            
            if stance_result['label'] != 'unclear':
                stance_edge = {
                    'speaker': 'author',  # Assuming author for now, quote detection will refine this
                    'target': entity_text,
                    'target_type': entity_type,
                    'label': stance_result['label'],
                    'score': stance_result['score'],
                    'method': stance_result['method'],
                    'evidence_spans': stance_result['evidence_spans'],
                    'text_preview': text[:100] + '...' if len(text) > 100 else text
                }
                stance_edges.append(stance_edge)
        
        return stance_edges

    def _hybrid_stance_classification(
        self, 
        text: str, 
        doc: spacy.tokens.Doc, 
        entity: str
    ) -> Dict[str, Any]:
        """Perform hybrid stance classification using both rules and MNLI.
        
        Args:
            text: Full message text
            doc: spaCy processed document
            entity: Target entity name
            
        Returns:
            Stance classification result
        """
        # Method 1: Dependency-based rules
        rules_result = self._dependency_rules_stance(doc, entity)
        
        # Method 2: Zero-shot MNLI
        mnli_result = self._mnli_stance(text, entity)
        
        # Combine results
        return self._combine_stance_results(rules_result, mnli_result, text, entity)

    def _dependency_rules_stance(
        self, 
        doc: spacy.tokens.Doc, 
        entity: str
    ) -> Dict[str, Any]:
        """Apply dependency parsing rules for stance detection.
        
        Args:
            doc: spaCy processed document
            entity: Target entity name
            
        Returns:
            Rules-based stance result
        """
        entity_lower = entity.lower()
        evidence_spans = []
        stance_signals = []
        
        # Find entity mentions in the document
        entity_tokens = []
        for token in doc:
            if entity_lower in token.text.lower():
                entity_tokens.append(token)
        
        # If no entity tokens found, try to find by substring
        if not entity_tokens:
            for sent in doc.sents:
                if entity_lower in sent.text.lower():
                    # Find approximate token positions
                    for token in sent:
                        if any(part in token.text.lower() for part in entity.lower().split()):
                            entity_tokens.append(token)
        
        # Analyze dependency relationships
        for entity_token in entity_tokens:
            # Look for stance verbs in the sentence
            sent = entity_token.sent
            
            for token in sent:
                if token.lemma_.lower() in self.support_verbs or token.lemma_.lower() in self.oppose_verbs:
                    # Check if verb is related to entity
                    if self._is_verb_related_to_entity(token, entity_token):
                        # Determine stance direction
                        base_stance = 'support' if token.lemma_.lower() in self.support_verbs else 'oppose'
                        
                        # Check for negations
                        is_negated = self._has_negation(token)
                        if is_negated:
                            base_stance = 'oppose' if base_stance == 'support' else 'support'
                        
                        # Check for hedges (reduce confidence)
                        has_hedge = self._has_hedge_words(sent)
                        confidence = 0.7 if has_hedge else 0.8
                        
                        stance_signals.append({
                            'stance': base_stance,
                            'confidence': confidence,
                            'verb': token.text,
                            'negated': is_negated,
                            'hedged': has_hedge
                        })
                        
                        # Add evidence span
                        evidence_spans.append({
                            'start': sent.start_char,
                            'end': sent.end_char,
                            'text': sent.text.strip(),
                            'span_type': 'dependency_rule',
                            'signal_verb': token.text
                        })
        
        # Aggregate signals
        if not stance_signals:
            return {
                'label': 'unclear',
                'score': 0.0,
                'method': 'rules',
                'evidence_spans': []
            }
        
        # Simple aggregation: majority vote weighted by confidence
        support_score = sum(s['confidence'] for s in stance_signals if s['stance'] == 'support')
        oppose_score = sum(s['confidence'] for s in stance_signals if s['stance'] == 'oppose')
        
        if support_score > oppose_score and support_score > 0.5:
            return {
                'label': 'support',
                'score': min(support_score, 1.0),
                'method': 'rules',
                'evidence_spans': evidence_spans
            }
        elif oppose_score > support_score and oppose_score > 0.5:
            return {
                'label': 'oppose',
                'score': min(oppose_score, 1.0),
                'method': 'rules',
                'evidence_spans': evidence_spans
            }
        else:
            return {
                'label': 'neutral',
                'score': 0.6,
                'method': 'rules',
                'evidence_spans': evidence_spans
            }

    def _is_verb_related_to_entity(
        self, 
        verb_token: spacy.tokens.Token, 
        entity_token: spacy.tokens.Token
    ) -> bool:
        """Check if a verb is syntactically related to an entity."""
        # Simple heuristics for dependency relationships
        # Check if entity is subject or object of verb
        if entity_token.head == verb_token:
            return True
        if verb_token.head == entity_token:
            return True
        
        # Check if they're in the same sentence and close by
        if (entity_token.sent == verb_token.sent and 
            abs(entity_token.i - verb_token.i) <= 5):
            return True
            
        return False

    def _has_negation(self, token: spacy.tokens.Token) -> bool:
        """Check if a token is negated."""
        # Check for negation in token's children
        for child in token.children:
            if child.dep_ == 'neg' or child.text.lower() in self.negative_words:
                return True
        
        # Check for negation in nearby tokens
        sent_tokens = list(token.sent)
        token_idx = sent_tokens.index(token)
        
        # Look for negation words within 3 tokens before
        for i in range(max(0, token_idx - 3), token_idx):
            if sent_tokens[i].text.lower() in self.negative_words:
                return True
                
        return False

    def _has_hedge_words(self, sent: spacy.tokens.Span) -> bool:
        """Check if sentence contains hedge words."""
        return any(token.text.lower() in self.hedge_words for token in sent)

    def _mnli_stance(self, text: str, entity: str) -> Dict[str, Any]:
        """Use zero-shot MNLI for stance classification.
        
        Args:
            text: Message text
            entity: Target entity
            
        Returns:
            MNLI-based stance result
        """
        try:
            # Create hypothesis templates
            hypotheses = [
                f"The author expresses support toward {entity}.",
                f"The author expresses opposition toward {entity}.",
                f"The author expresses neutral sentiment toward {entity}."
            ]
            
            # Get predictions
            result = self.mnli_pipeline(text, hypotheses)
            
            # Map labels to standard format
            label_mapping = {
                f"The author expresses support toward {entity}.": "support",
                f"The author expresses opposition toward {entity}.": "oppose",
                f"The author expresses neutral sentiment toward {entity}.": "neutral"
            }
            
            best_label = result['labels'][0]
            best_score = result['scores'][0]
            
            mapped_label = label_mapping.get(best_label, 'unclear')
            
            return {
                'label': mapped_label,
                'score': float(best_score),
                'method': 'mnli',
                'evidence_spans': [{
                    'start': 0,
                    'end': len(text),
                    'text': text,
                    'span_type': 'mnli_classification',
                    'hypothesis': best_label
                }]
            }
            
        except Exception as e:
            logger.warning(f"MNLI classification failed for entity '{entity}': {e}")
            return {
                'label': 'unclear',
                'score': 0.0,
                'method': 'mnli',
                'evidence_spans': []
            }

    def _combine_stance_results(
        self, 
        rules_result: Dict[str, Any],
        mnli_result: Dict[str, Any],
        text: str,
        entity: str
    ) -> Dict[str, Any]:
        """Combine results from dependency rules and MNLI.
        
        Args:
            rules_result: Result from dependency rules
            mnli_result: Result from MNLI
            text: Original text
            entity: Target entity
            
        Returns:
            Combined stance result
        """
        # If both methods agree and have high confidence
        if (rules_result['label'] == mnli_result['label'] and 
            rules_result['label'] != 'unclear' and
            rules_result['score'] > 0.6 and mnli_result['score'] > 0.6):
            
            # High confidence combined result
            combined_score = (
                rules_result['score'] * self.dep_rules_weight + 
                mnli_result['score'] * self.mnli_weight
            )
            
            return {
                'label': rules_result['label'],
                'score': min(combined_score, 1.0),
                'method': 'hybrid_agree',
                'evidence_spans': rules_result['evidence_spans'] + mnli_result['evidence_spans']
            }
        
        # If rules have high confidence, prefer rules (more interpretable)
        elif rules_result['label'] != 'unclear' and rules_result['score'] > 0.7:
            return {
                'label': rules_result['label'],
                'score': rules_result['score'],
                'method': 'rules_preferred',
                'evidence_spans': rules_result['evidence_spans']
            }
        
        # If MNLI has high confidence and rules are unclear
        elif mnli_result['label'] != 'unclear' and mnli_result['score'] > 0.8:
            return {
                'label': mnli_result['label'],
                'score': mnli_result['score'],
                'method': 'mnli_preferred',
                'evidence_spans': mnli_result['evidence_spans']
            }
        
        # Otherwise, default to unclear (neutral by design)
        else:
            return {
                'label': 'unclear',
                'score': 0.0,
                'method': 'hybrid_unclear',
                'evidence_spans': []
            }

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process dataframe and add stance information.
        
        Args:
            df: DataFrame with text data and entities
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added stance columns
        """
        logger.info(f"Processing {len(df)} messages for stance classification")
        
        if "entities" not in df.columns:
            logger.warning("No entities column found. Run NER processor first.")
            df["stance_edges"] = [[] for _ in range(len(df))]
            return df
        
        all_stance_edges = []
        
        for i, row in df.iterrows():
            try:
                text = row[text_column]
                entities = row["entities"] or []
                
                # Classify stance for this message
                stance_edges = self.classify_stance(text, entities)
                all_stance_edges.append(stance_edges)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1} messages")
                    
            except Exception as e:
                logger.warning(f"Failed to classify stance for message {i}: {e}")
                all_stance_edges.append([])
        
        # Add stance edges to dataframe
        df = df.copy()
        df["stance_edges"] = all_stance_edges
        
        # Add summary statistics
        df["stance_edge_count"] = df["stance_edges"].apply(len)
        
        # Log summary
        total_edges = sum(len(edges) for edges in all_stance_edges)
        if total_edges > 0:
            # Count by label
            all_edges = [edge for edges in all_stance_edges for edge in edges]
            label_counts = {}
            for edge in all_edges:
                label = edge['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            logger.info(f"Stance classification completed.")
            logger.info(f"Total stance edges: {total_edges}")
            logger.info(f"Stance distribution: {label_counts}")
            logger.info(f"Average edges per message: {total_edges/len(df):.2f}")
        
        return df

    def get_stance_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about stance classification results.
        
        Args:
            df: DataFrame with stance classification results
            
        Returns:
            Dictionary with stance statistics
        """
        if "stance_edges" not in df.columns:
            return {}

        # Flatten all stance edges
        all_edges = []
        for edges in df["stance_edges"]:
            all_edges.extend(edges)

        if not all_edges:
            return {
                "total_stance_edges": 0,
                "messages_with_stance": 0,
            }

        # Count by label
        label_counts = {}
        method_counts = {}
        target_counts = {}
        
        for edge in all_edges:
            label = edge['label']
            method = edge['method']
            target = edge['target']
            
            label_counts[label] = label_counts.get(label, 0) + 1
            method_counts[method] = method_counts.get(method, 0) + 1
            target_counts[target] = target_counts.get(target, 0) + 1

        # Get top targets
        top_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_stance_edges": len(all_edges),
            "messages_with_stance": sum(1 for edges in df["stance_edges"] if edges),
            "stance_distribution": label_counts,
            "method_distribution": method_counts,
            "top_targets": dict(top_targets),
            "avg_confidence": sum(edge['score'] for edge in all_edges) / len(all_edges),
        }