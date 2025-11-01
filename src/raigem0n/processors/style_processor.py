"""Style features processor for linguistic pattern extraction."""

import logging
import re
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class StyleProcessor:
    """Extract stylistic and linguistic features from text."""

    def __init__(self) -> None:
        """Initialize style processor."""
        
        # Hedge words (uncertainty markers)
        self.hedge_words = {
            'might', 'maybe', 'perhaps', 'possibly', 'seems', 'appears',
            'probably', 'likely', 'could', 'would', 'should', 'allegedly',
            'supposedly', 'presumably', 'apparently', 'potentially',
            'conceivably', 'plausibly', 'ostensibly'
        }
        
        # Superlatives (intensifiers)
        self.superlatives = {
            'best', 'worst', 'greatest', 'largest', 'smallest', 'highest', 'lowest',
            'most', 'least', 'ever', 'never', 'always', 'completely', 'totally',
            'absolutely', 'extremely', 'incredibly', 'amazing', 'terrible',
            'perfect', 'horrible', 'fantastic', 'awful', 'brilliant', 'stupid',
            'genius', 'disaster', 'epic', 'massive', 'huge', 'tiny'
        }
        
        # Emotion words
        self.emotion_words = {
            'love', 'hate', 'angry', 'furious', 'excited', 'thrilled',
            'disappointed', 'frustrated', 'happy', 'sad', 'scared',
            'worried', 'concerned', 'optimistic', 'pessimistic', 'hope',
            'fear', 'joy', 'rage', 'disgust', 'surprise', 'trust'
        }
        
        # Question patterns
        self.question_patterns = [
            r'\?',                          # Direct question marks
            r'\bwhy\s+(?:is|are|do|does|did|would|should|can|could)',
            r'\bwhat\s+(?:is|are|do|does|did|would|should|can|could)',
            r'\bhow\s+(?:is|are|do|does|did|would|should|can|could)',
            r'\bwhen\s+(?:is|are|do|does|did|would|should|can|could)',
            r'\bwhere\s+(?:is|are|do|does|did|would|should|can|could)',
            r'\bwho\s+(?:is|are|do|does|did|would|should|can|could)',
        ]
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Emoji pattern (basic)
        self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]')

    def extract_style_features(self, text: str) -> Dict[str, Any]:
        """Extract stylistic features from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of style features
        """
        if not text or len(text.strip()) == 0:
            return self._empty_features()
        
        text_clean = text.strip()
        words = text_clean.lower().split()
        sentences = self._split_sentences(text_clean)
        
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text_clean)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['period_count'] = text.count('.')
        features['comma_count'] = text.count(',')
        features['semicolon_count'] = text.count(';')
        features['colon_count'] = text.count(':')
        features['dash_count'] = text.count('—') + text.count('--')
        
        # Capitalization features
        features['all_caps_count'] = sum(1 for word in words if word.isupper() and len(word) > 1)
        features['caps_ratio'] = features['all_caps_count'] / len(words) if words else 0
        features['title_case_count'] = sum(1 for word in words if word.istitle())
        
        # Linguistic style features
        features['hedge_word_count'] = sum(1 for word in words if word in self.hedge_words)
        features['superlative_count'] = sum(1 for word in words if word in self.superlatives)
        features['emotion_word_count'] = sum(1 for word in words if word in self.emotion_words)
        
        # Question and rhetorical features
        features['direct_questions'] = self._count_questions(text_clean)
        features['rhetorical_indicators'] = self._count_rhetorical_patterns(text_clean)
        
        # Repetition and emphasis
        features['repeated_punctuation'] = self._count_repeated_punctuation(text_clean)
        features['word_repetition'] = self._calculate_word_repetition(words)
        
        # URLs and mentions
        features['url_count'] = len(self.url_pattern.findall(text))
        features['mention_count'] = len(re.findall(r'@\w+', text))
        features['hashtag_count'] = len(re.findall(r'#\w+', text))
        
        # Emojis (basic detection)
        features['emoji_count'] = len(self.emoji_pattern.findall(text))
        
        # Readability approximation (Flesch-like)
        features['readability_score'] = self._calculate_readability(
            features['word_count'],
            features['sentence_count'],
            features['avg_word_length']
        )
        
        # Intensity score (combination of various emphasis markers)
        features['intensity_score'] = self._calculate_intensity_score(features)
        
        # Formality indicators
        features['formality_score'] = self._calculate_formality_score(text_clean, words)
        
        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty features dictionary."""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0.0, 'avg_sentence_length': 0.0,
            'exclamation_count': 0, 'question_count': 0, 'period_count': 0,
            'comma_count': 0, 'semicolon_count': 0, 'colon_count': 0,
            'dash_count': 0, 'all_caps_count': 0, 'caps_ratio': 0.0,
            'title_case_count': 0, 'hedge_word_count': 0, 'superlative_count': 0,
            'emotion_word_count': 0, 'direct_questions': 0, 'rhetorical_indicators': 0,
            'repeated_punctuation': 0, 'word_repetition': 0.0, 'url_count': 0,
            'mention_count': 0, 'hashtag_count': 0, 'emoji_count': 0,
            'readability_score': 0.0, 'intensity_score': 0.0, 'formality_score': 0.0
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_questions(self, text: str) -> int:
        """Count question patterns in text."""
        question_count = 0
        for pattern in self.question_patterns:
            question_count += len(re.findall(pattern, text, re.IGNORECASE))
        return question_count

    def _count_rhetorical_patterns(self, text: str) -> int:
        """Count rhetorical patterns and discourse markers."""
        rhetorical_patterns = [
            r'\b(?:think about it|let me be clear|the fact is|here\'s the thing)',
            r'\b(?:don\'t you think|wouldn\'t you agree|right\?)',
            r'\b(?:obviously|clearly|undoubtedly|without question)',
            r'🤔',  # thinking emoji often used rhetorically
        ]
        
        count = 0
        for pattern in rhetorical_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _count_repeated_punctuation(self, text: str) -> int:
        """Count instances of repeated punctuation for emphasis."""
        repeated_patterns = [
            r'!{2,}',    # !!!, !!!!, etc.
            r'\?{2,}',   # ???, ????, etc.
            r'\.{3,}',   # ..., ...., etc.
        ]
        
        count = 0
        for pattern in repeated_patterns:
            count += len(re.findall(pattern, text))
        return count

    def _calculate_word_repetition(self, words: List[str]) -> float:
        """Calculate degree of word repetition."""
        if len(words) <= 1:
            return 0.0
        
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count substantial words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate repetition ratio
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        return repeated_words / len(words) if words else 0.0

    def _calculate_readability(self, word_count: int, sentence_count: int, avg_word_length: float) -> float:
        """Calculate approximate readability score (0-100, higher = more readable)."""
        if sentence_count == 0 or word_count == 0:
            return 0.0
        
        # Simplified Flesch-like formula
        avg_sentence_length = word_count / sentence_count
        
        # Invert the formula so higher = more readable
        score = 100 - (1.015 * avg_sentence_length + 84.6 * (avg_word_length / 4.7))
        
        # Clamp to 0-100 range
        return max(0.0, min(100.0, score))

    def _calculate_intensity_score(self, features: Dict[str, Any]) -> float:
        """Calculate overall intensity/emphasis score."""
        # Combine various emphasis indicators
        intensity_factors = [
            features['exclamation_count'] * 0.3,
            features['all_caps_count'] * 0.4,
            features['superlative_count'] * 0.2,
            features['emotion_word_count'] * 0.2,
            features['repeated_punctuation'] * 0.5,
        ]
        
        raw_score = sum(intensity_factors)
        word_count = features['word_count']
        
        # Normalize by word count to get intensity per word
        normalized_score = (raw_score / word_count) * 10 if word_count > 0 else 0
        
        # Cap at reasonable maximum
        return min(normalized_score, 10.0)

    def _calculate_formality_score(self, text: str, words: List[str]) -> float:
        """Calculate formality score (0-10, higher = more formal)."""
        if not words:
            return 0.0
        
        formal_indicators = 0
        informal_indicators = 0
        
        # Formal indicators
        formal_patterns = [
            r'\b(?:furthermore|moreover|nevertheless|consequently|therefore|thus)\b',
            r'\b(?:pursuant|regarding|aforementioned|heretofore|whereas)\b',
            r'\b(?:indicate|demonstrate|establish|constitute|implement)\b',
        ]
        
        for pattern in formal_patterns:
            formal_indicators += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Informal indicators
        informal_patterns = [
            r'\b(?:gonna|wanna|gotta|kinda|sorta|yeah|ok|okay)\b',
            r'\b(?:awesome|cool|sucks|crap|damn|hell)\b',
            r'[!]{2,}',  # Multiple exclamations
        ]
        
        for pattern in informal_patterns:
            informal_indicators += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Contractions are informal
        contractions = len(re.findall(r"\w+'\w+", text))
        informal_indicators += contractions
        
        # Calculate formality (0-10 scale)
        total_indicators = formal_indicators + informal_indicators
        if total_indicators == 0:
            return 5.0  # Neutral
        
        formality_ratio = formal_indicators / total_indicators
        return formality_ratio * 10

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process dataframe and add style features.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added style feature columns
        """
        logger.info(f"Processing {len(df)} messages for style feature extraction")
        
        all_features = []
        
        for i, row in df.iterrows():
            try:
                text = row[text_column]
                features = self.extract_style_features(text)
                all_features.append(features)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} messages")
                    
            except Exception as e:
                logger.warning(f"Failed to extract style features for message {i}: {e}")
                all_features.append(self._empty_features())
        
        # Add features to dataframe
        df = df.copy()
        
        # Add style features as a JSON column
        df["style_features"] = all_features
        
        # Add some key features as individual columns for easier analysis
        df["exclamation_count"] = [f["exclamation_count"] for f in all_features]
        df["caps_ratio"] = [f["caps_ratio"] for f in all_features]
        df["hedge_word_count"] = [f["hedge_word_count"] for f in all_features]
        df["superlative_count"] = [f["superlative_count"] for f in all_features]
        df["intensity_score"] = [f["intensity_score"] for f in all_features]
        df["formality_score"] = [f["formality_score"] for f in all_features]
        df["readability_score"] = [f["readability_score"] for f in all_features]
        
        # Log summary statistics
        logger.info(f"Style feature extraction completed.")
        if len(all_features) > 0 and len(df) > 0:
            avg_intensity = sum(f["intensity_score"] for f in all_features) / len(all_features)
            avg_formality = sum(f["formality_score"] for f in all_features) / len(all_features)
            avg_readability = sum(f["readability_score"] for f in all_features) / len(all_features)
            high_intensity_count = sum(1 for f in all_features if f["intensity_score"] > 5.0)

            logger.info(f"Average intensity score: {avg_intensity:.2f}")
            logger.info(f"Average formality score: {avg_formality:.2f}")
            logger.info(f"Average readability score: {avg_readability:.2f}")
            logger.info(f"High intensity messages: {high_intensity_count}/{len(df)} ({high_intensity_count/len(df)*100:.1f}%)")
        else:
            logger.info("No features extracted")
        
        return df

    def get_style_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about style features.
        
        Args:
            df: DataFrame with style feature results
            
        Returns:
            Dictionary with style statistics
        """
        if "style_features" not in df.columns:
            return {}

        all_features = df["style_features"].tolist()
        
        if not all_features:
            return {}

        # Aggregate statistics
        stats = {}
        
        # Calculate means for numeric features
        numeric_features = [
            'exclamation_count', 'caps_ratio', 'hedge_word_count', 'superlative_count',
            'intensity_score', 'formality_score', 'readability_score',
            'word_count', 'sentence_count', 'emoji_count', 'url_count'
        ]
        
        for feature in numeric_features:
            values = [f.get(feature, 0) for f in all_features]
            stats[f'avg_{feature}'] = sum(values) / len(values) if values else 0
            stats[f'max_{feature}'] = max(values) if values else 0
        
        # Distribution statistics
        stats['high_intensity_messages'] = sum(1 for f in all_features if f.get('intensity_score', 0) > 5.0)
        stats['formal_messages'] = sum(1 for f in all_features if f.get('formality_score', 0) > 7.0)
        stats['informal_messages'] = sum(1 for f in all_features if f.get('formality_score', 0) < 3.0)
        stats['high_caps_messages'] = sum(1 for f in all_features if f.get('caps_ratio', 0) > 0.1)
        
        return stats

    def export_style_features_json(self, df: pd.DataFrame, output_path: str) -> None:
        """Export style features to JSON file.
        
        Args:
            df: DataFrame with style feature results
            output_path: Path to save the JSON file
        """
        import json
        from pathlib import Path
        
        if "style_features" not in df.columns:
            logger.warning("No style features found in dataframe")
            return
            
        output_path = Path(output_path)
        
        # Create summary structure
        style_data = {
            "metadata": {
                "total_messages": len(df),
                "extraction_date": pd.Timestamp.now().isoformat(),
                "processor_version": "1.0.0"
            },
            "summary_stats": self.get_style_stats(df),
            "message_features": []
        }
        
        # Add per-message features
        for idx, row in df.iterrows():
            message_data = {
                "msg_id": str(row.get("msg_id", idx)),
                "date": row.get("date").isoformat() if pd.notna(row.get("date")) else None,
                "text_length": len(row.get("text", "")),
                "style_features": row["style_features"]
            }
            style_data["message_features"].append(message_data)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(style_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Style features exported to {output_path}")
        logger.info(f"Exported {len(style_data['message_features'])} message feature sets")
