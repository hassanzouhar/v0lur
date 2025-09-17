"""Named Entity Recognition processor with entity aliasing."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


logger = logging.getLogger(__name__)


class NERProcessor:
    """Named Entity Recognition with entity aliasing."""

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        max_entities_per_msg: int = 3,
        device: Optional[str] = None,
    ) -> None:
        """Initialize NER processor."""
        self.model_name = model_name
        self.max_entities_per_msg = max_entities_per_msg
        self.device = device
        self.ner_pipeline = None
        self.aliases: Dict[str, Dict[str, Any]] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load NER model."""
        try:
            logger.info(f"Loading NER model: {self.model_name}")
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            raise

    def load_aliases(self, aliases: Dict[str, Dict[str, Any]]) -> None:
        """Load entity aliases for canonical mapping."""
        self.aliases = aliases
        logger.info(f"Loaded {len(aliases)} entity aliases")

        # Create reverse lookup for aliases with type information
        self._alias_lookup = {}
        for canonical, config in aliases.items():
            # Add canonical name with type info
            self._alias_lookup[canonical.lower()] = {
                "canonical": canonical,
                "type": config.get("type", "MISC")
            }
            # Add all aliases with type info
            for alias in config.get("aliases", []):
                self._alias_lookup[alias.lower()] = {
                    "canonical": canonical,
                    "type": config.get("type", "MISC")
                }

    def extract_entities(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract entities from texts."""
        if not self.ner_pipeline:
            raise RuntimeError("NER model not loaded")

        logger.debug(f"Extracting entities from {len(texts)} texts")
        all_entities = []

        for text in texts:
            try:
                # Extract raw entities
                raw_entities = self.ner_pipeline(text)
                
                # Process and filter entities
                processed_entities = self._process_entities(raw_entities, text)
                all_entities.append(processed_entities)

            except Exception as e:
                logger.warning(f"Failed to extract entities from text: {e}")
                all_entities.append([])

        return all_entities

    def _process_entities(
        self, raw_entities: List[Dict[str, Any]], original_text: str
    ) -> List[Dict[str, Any]]:
        """Process and clean raw entity extractions."""
        processed = []
        seen_entities = set()

        for entity in raw_entities:
            # Get entity text
            entity_text = entity.get("word", "").strip()
            if not entity_text:
                continue

            # Clean entity text (remove special tokens)
            entity_text = entity_text.replace("##", "")
            
            # Skip very short or clearly invalid entities
            if len(entity_text) < 2 or entity_text.isdigit():
                continue
            
            # Skip common garbage entities that are often NER errors
            garbage_entities = {
                "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "an", "a",
                "anti", "pro", "non", "pre", "post", "de", "la", "el", "le", "les", "der", "die", "das",
                "rep", "sen", "mr", "mrs", "ms", "dr", "prof", "jr", "sr", "ii", "iii", "iv", "show"
            }
            if entity_text.lower() in garbage_entities:
                continue
            
            # Skip entities that are clearly partial words or tokens
            if entity_text.lower() in {"iden", "ka", "po", "se", "ch", "charlie", "de"}:
                continue

            # Get initial entity type from NER model
            initial_entity_type = self._normalize_entity_type(entity.get("entity_group", "MISC"))
            
            # Apply canonical mapping (this may override the entity type)
            canonical_entity, entity_type = self._get_canonical_entity(entity_text, initial_entity_type)
            
            # Avoid duplicates (case-insensitive)
            entity_key = f"{canonical_entity.lower()}:{entity_type}"
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)

            # Create processed entity
            processed_entity = {
                "text": canonical_entity,
                "type": entity_type,
                "confidence": float(entity.get("score", 0.0)),
                "start": int(entity.get("start", 0)),
                "end": int(entity.get("end", len(original_text))),
                "original_text": entity_text,
            }
            
            processed.append(processed_entity)

        # Sort by confidence and limit count
        processed.sort(key=lambda x: x["confidence"], reverse=True)
        return processed[: self.max_entities_per_msg]

    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity types to standard format."""
        entity_type = entity_type.upper()
        
        # Map various entity types to our standard set
        type_mapping = {
            "PER": "PERSON",
            "PERSON": "PERSON",
            "ORG": "ORG", 
            "ORGANIZATION": "ORG",
            "LOC": "LOC",
            "LOCATION": "LOC",
            "GPE": "LOC",  # Geopolitical entity
            "MISC": "MISC",
            "MISCELLANEOUS": "MISC",
        }
        
        return type_mapping.get(entity_type, "MISC")

    def _get_canonical_entity(self, entity_text: str, entity_type: str) -> Tuple[str, str]:
        """Map entity text to canonical form using aliases.
        
        Returns:
            Tuple[str, str]: (canonical_entity_name, final_entity_type)
        """
        if not self.aliases:
            return entity_text, entity_type

        # Direct lookup
        alias_info = self._alias_lookup.get(entity_text.lower())
        if alias_info:
            return alias_info["canonical"], alias_info["type"]

        # Fuzzy matching for person names (simple approach)
        if entity_type == "PERSON":
            # Try last name matching for persons
            entity_parts = entity_text.split()
            if len(entity_parts) > 1:
                last_name = entity_parts[-1].lower()
                for alias_key, alias_info in self._alias_lookup.items():
                    if last_name in alias_key.split():
                        return alias_info["canonical"], alias_info["type"]

        return entity_text, entity_type

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process dataframe and add entity information."""
        logger.info(f"Processing {len(df)} messages for entity extraction")
        
        # Extract entities in batches
        batch_size = 50  # Reasonable batch size for memory management
        all_entities = []
        
        for i in range(0, len(df), batch_size):
            batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
            batch_entities = self.extract_entities(batch_texts)
            all_entities.extend(batch_entities)
            
            if (i + batch_size) % 200 == 0:
                logger.info(f"Processed {min(i + batch_size, len(df))} messages")

        # Add entities to dataframe
        df = df.copy()
        df["entities"] = all_entities
        
        # Add entity summary columns
        df["entity_count"] = df["entities"].apply(len)
        df["person_entities"] = df["entities"].apply(
            lambda ents: [e["text"] for e in ents if e["type"] == "PERSON"]
        )
        df["org_entities"] = df["entities"].apply(
            lambda ents: [e["text"] for e in ents if e["type"] == "ORG"]  
        )
        df["loc_entities"] = df["entities"].apply(
            lambda ents: [e["text"] for e in ents if e["type"] == "LOC"]
        )

        logger.info(f"Entity extraction completed. Average entities per message: {df['entity_count'].mean():.2f}")
        return df

    def get_entity_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about extracted entities."""
        if "entities" not in df.columns:
            return {}

        all_entities = []
        for entities_list in df["entities"]:
            all_entities.extend(entities_list)

        if not all_entities:
            return {"total_entities": 0}

        # Count by type
        type_counts = {}
        entity_counts = {}
        
        for entity in all_entities:
            entity_type = entity["type"]
            entity_text = entity["text"]
            
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            entity_counts[entity_text] = entity_counts.get(entity_text, 0) + 1

        # Get top entities
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_entities": len(all_entities),
            "unique_entities": len(entity_counts),
            "entities_by_type": type_counts,
            "top_entities": dict(top_entities),
            "avg_entities_per_message": len(all_entities) / len(df) if len(df) > 0 else 0,
        }