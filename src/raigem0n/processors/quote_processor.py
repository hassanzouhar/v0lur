"""Quote detection and span tagging processor for attribution accuracy."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class QuoteProcessor:
    """Detect and tag quoted/forwarded content for proper speaker attribution."""

    def __init__(
        self,
        detect_forwarded: bool = True,
        detect_quoted_spans: bool = True,
        max_quote_length: int = 2048,
        attribute_forwarded_to_source: bool = True,
    ) -> None:
        """Initialize quote processor.
        
        Args:
            detect_forwarded: Whether to detect forwarded messages
            detect_quoted_spans: Whether to detect quoted text spans
            max_quote_length: Maximum length of quoted text to process
            attribute_forwarded_to_source: Whether to attribute forwarded content to source
        """
        self.detect_forwarded = detect_forwarded
        self.detect_quoted_spans = detect_quoted_spans
        self.max_quote_length = max_quote_length
        self.attribute_forwarded_to_source = attribute_forwarded_to_source
        
        # Quote detection patterns
        self.quote_patterns = [
            # Typographic quotes
            r'"([^"]{10,})"',                    # "quoted text"
            r'"([^"]{10,})"',                    # "quoted text" (smart quotes)
            r'\u2018([^\u2019]{10,})\u2019',        # 'quoted text' (smart quotes)
            
            # Block quotes with prefix
            r'^>\s*(.+)$',                       # > quoted line
            r'^\|\s*(.+)$',                      # | quoted line
            
            # Attribution markers
            r'—\s*(.+)$',                        # — attribution
            r'--\s*(.+)$',                       # -- attribution
            r'\n\s*-\s*([A-Z][^.!?]*[.!?])$',    # - Attribution at end
        ]
        
        # Forwarded message patterns
        self.forwarded_patterns = [
            r'forwarded\s+from\s+(.+?)[:：]',    # Forwarded from X:
            r'forwarded\s+message',              # Forwarded message
            r'fwd:\s*',                          # Fwd:
            r're:\s*',                           # Re: (sometimes forwarded)
        ]
        
        # Speaker attribution patterns
        self.speaker_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+said[:\s]',     # John Doe said:
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+tweeted[:\s]',  # John Doe tweeted:
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+wrote[:\s]',    # John Doe wrote:
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+stated[:\s]',   # John Doe stated:
            r'according\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # according to X
            r'as\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+put\s+it',   # as X put it
        ]

    def detect_quotes_and_spans(
        self, 
        text: str,
        forwarded_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect quotes and tag spans in text.
        
        Args:
            text: Message text to analyze
            forwarded_from: Optional forwarded source information
            
        Returns:
            Dictionary with span analysis results
        """
        if not text or len(text.strip()) == 0:
            return self._empty_result()
            
        spans = []
        speakers = set()
        
        # Check for forwarded message
        if self.detect_forwarded and (forwarded_from or self._is_forwarded_message(text)):
            forwarded_info = self._process_forwarded_message(text, forwarded_from)
            spans.extend(forwarded_info['spans'])
            speakers.update(forwarded_info['speakers'])
        
        # Detect quoted spans in text
        if self.detect_quoted_spans:
            quoted_info = self._detect_quoted_spans(text)
            spans.extend(quoted_info['spans'])
            speakers.update(quoted_info['speakers'])
        
        # If no special spans detected, tag everything as author
        if not spans:
            spans = [{
                'start': 0,
                'end': len(text),
                'text': text,
                'span_type': 'author',
                'speaker': 'author',
                'confidence': 1.0
            }]
        else:
            # Fill gaps with author spans
            spans = self._fill_author_gaps(text, spans)
        
        # Sort spans by start position
        spans.sort(key=lambda x: x['start'])
        
        return {
            'spans': spans,
            'speakers': list(speakers),
            'has_quotes': any(s['span_type'] in ['quoted', 'forwarded'] for s in spans),
            'multi_speaker': len(speakers) > 1,
            'total_spans': len(spans)
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'spans': [],
            'speakers': [],
            'has_quotes': False,
            'multi_speaker': False,
            'total_spans': 0
        }

    def _is_forwarded_message(self, text: str) -> bool:
        """Check if message appears to be forwarded."""
        text_lower = text.lower()
        
        for pattern in self.forwarded_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                return True
        
        return False

    def _process_forwarded_message(
        self, 
        text: str, 
        forwarded_from: Optional[str]
    ) -> Dict[str, Any]:
        """Process forwarded message content."""
        spans = []
        speakers = set()
        
        if forwarded_from:
            # Use provided forwarded source
            speaker = self._clean_speaker_name(forwarded_from)
            speakers.add(speaker)
            
            spans.append({
                'start': 0,
                'end': len(text),
                'text': text,
                'span_type': 'forwarded',
                'speaker': speaker,
                'confidence': 0.9,
                'source': forwarded_from
            })
        else:
            # Try to extract forwarded source from text
            for pattern in self.forwarded_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    if match.groups():
                        speaker = self._clean_speaker_name(match.group(1))
                        speakers.add(speaker)
                        
                        # Try to find the actual forwarded content
                        forwarded_start = match.end()
                        forwarded_text = text[forwarded_start:].strip()
                        
                        if forwarded_text:
                            spans.append({
                                'start': forwarded_start,
                                'end': len(text),
                                'text': forwarded_text,
                                'span_type': 'forwarded',
                                'speaker': speaker,
                                'confidence': 0.8
                            })
                    break
        
        return {
            'spans': spans,
            'speakers': speakers
        }

    def _detect_quoted_spans(self, text: str) -> Dict[str, Any]:
        """Detect quoted text spans and their speakers."""
        spans = []
        speakers = set()
        
        # Look for direct quotes with attribution
        for speaker_pattern in self.speaker_patterns:
            for match in re.finditer(speaker_pattern, text, re.IGNORECASE):
                speaker = self._clean_speaker_name(match.group(1))
                speakers.add(speaker)
                
                # Look for quoted content after the attribution
                remaining_text = text[match.end():]
                quote_match = self._find_quote_after_attribution(remaining_text)
                
                if quote_match:
                    quote_start = match.end() + quote_match['start']
                    quote_end = match.end() + quote_match['end']
                    
                    spans.append({
                        'start': quote_start,
                        'end': quote_end,
                        'text': quote_match['text'],
                        'span_type': 'quoted',
                        'speaker': speaker,
                        'confidence': 0.8,
                        'attribution_pattern': speaker_pattern
                    })
        
        # Look for standalone quotes (without clear attribution)
        for quote_pattern in self.quote_patterns:
            for match in re.finditer(quote_pattern, text, re.MULTILINE):
                quote_text = match.group(1) if match.groups() else match.group(0)
                
                # Skip if this quote was already attributed
                if any(span['start'] <= match.start() < span['end'] for span in spans):
                    continue
                
                # Skip very short quotes
                if len(quote_text.strip()) < 10:
                    continue
                
                spans.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': quote_text.strip(),
                    'span_type': 'quoted',
                    'speaker': 'unknown',
                    'confidence': 0.6,
                    'attribution_pattern': quote_pattern
                })
                
                speakers.add('unknown')
        
        return {
            'spans': spans,
            'speakers': speakers
        }

    def _find_quote_after_attribution(self, text: str) -> Optional[Dict[str, Any]]:
        """Find quoted content after speaker attribution."""
        text = text.strip()
        if not text:
            return None
        
        # Look for quotes starting immediately after attribution
        for quote_pattern in self.quote_patterns:
            match = re.search(quote_pattern, text)
            if match and match.start() < 50:  # Quote should start soon after attribution
                quote_text = match.group(1) if match.groups() else match.group(0)
                return {
                    'start': match.start(),
                    'end': match.end(),
                    'text': quote_text.strip()
                }
        
        # If no explicit quotes, check if the whole remaining text might be a quote
        if len(text) < self.max_quote_length and self._looks_like_quote(text):
            return {
                'start': 0,
                'end': len(text),
                'text': text
            }
        
        return None

    def _looks_like_quote(self, text: str) -> bool:
        """Heuristic to determine if text looks like quoted content."""
        # Check for quote-like characteristics
        quote_indicators = [
            text.startswith('"') or text.startswith('"'),
            text.endswith('"') or text.endswith('"'),
            'I ' in text and text.count('I ') >= 2,  # First person
            any(word in text.lower() for word in ['my ', 'our ', 'we ', 'us ']),
            text.count('!') >= 2,  # Emotional content
        ]
        
        return sum(quote_indicators) >= 2

    def _clean_speaker_name(self, name: str) -> str:
        """Clean and normalize speaker name."""
        if not name:
            return 'unknown'
        
        # Remove common prefixes/suffixes
        name = re.sub(r'^(from|by|@)\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*(said|wrote|tweeted|stated).*$', '', name, flags=re.IGNORECASE)
        
        # Clean up whitespace and punctuation
        name = re.sub(r'[^\w\s\-\.]', '', name)
        name = ' '.join(name.split())
        
        return name.strip() or 'unknown'

    def _fill_author_gaps(self, text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fill gaps between spans with author attribution."""
        if not spans:
            return spans
        
        filled_spans = []
        current_pos = 0
        
        for span in sorted(spans, key=lambda x: x['start']):
            # Add author span for gap before this span
            if current_pos < span['start']:
                gap_text = text[current_pos:span['start']].strip()
                if gap_text:
                    filled_spans.append({
                        'start': current_pos,
                        'end': span['start'],
                        'text': gap_text,
                        'span_type': 'author',
                        'speaker': 'author',
                        'confidence': 1.0
                    })
            
            filled_spans.append(span)
            current_pos = max(current_pos, span['end'])
        
        # Add final author span if there's remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                filled_spans.append({
                    'start': current_pos,
                    'end': len(text),
                    'text': remaining_text,
                    'span_type': 'author',
                    'speaker': 'author',
                    'confidence': 1.0
                })
        
        return filled_spans

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process dataframe and add quote/span information.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added quote/span columns
        """
        logger.info(f"Processing {len(df)} messages for quote detection")
        
        all_results = []
        
        for i, row in df.iterrows():
            try:
                text = row[text_column]
                forwarded_from = row.get('forwarded_from', None)
                
                # Detect quotes and spans
                result = self.detect_quotes_and_spans(text, forwarded_from)
                all_results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} messages")
                    
            except Exception as e:
                logger.warning(f"Failed to process quotes for message {i}: {e}")
                all_results.append(self._empty_result())
        
        # Add results to dataframe
        df = df.copy()
        df["quote_spans"] = [result["spans"] for result in all_results]
        df["quote_speakers"] = [result["speakers"] for result in all_results]
        df["has_quotes"] = [result["has_quotes"] for result in all_results]
        df["multi_speaker"] = [result["multi_speaker"] for result in all_results]
        df["span_count"] = [result["total_spans"] for result in all_results]
        
        # Log summary
        messages_with_quotes = sum(result["has_quotes"] for result in all_results)
        multi_speaker_messages = sum(result["multi_speaker"] for result in all_results)
        total_spans = sum(result["total_spans"] for result in all_results)

        logger.info(f"Quote detection completed.")
        if len(df) > 0:
            logger.info(f"Messages with quotes: {messages_with_quotes}/{len(df)} ({messages_with_quotes/len(df)*100:.1f}%)")
            logger.info(f"Multi-speaker messages: {multi_speaker_messages}/{len(df)} ({multi_speaker_messages/len(df)*100:.1f}%)")
            logger.info(f"Average spans per message: {total_spans/len(df):.2f}")
        else:
            logger.info("No messages processed")
        
        return df

    def get_quote_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about quote detection results.
        
        Args:
            df: DataFrame with quote detection results
            
        Returns:
            Dictionary with quote statistics
        """
        if "quote_spans" not in df.columns:
            return {}

        # Aggregate statistics
        messages_with_quotes = df["has_quotes"].sum() if "has_quotes" in df.columns else 0
        multi_speaker_messages = df["multi_speaker"].sum() if "multi_speaker" in df.columns else 0
        total_spans = df["span_count"].sum() if "span_count" in df.columns else 0
        
        # Count span types
        span_type_counts = {}
        all_speakers = set()
        
        for spans in df["quote_spans"]:
            for span in spans:
                span_type = span.get("span_type", "unknown")
                span_type_counts[span_type] = span_type_counts.get(span_type, 0) + 1
                
                speaker = span.get("speaker", "unknown")
                if speaker != "author":
                    all_speakers.add(speaker)
        
        return {
            "total_messages": len(df),
            "messages_with_quotes": int(messages_with_quotes),
            "multi_speaker_messages": int(multi_speaker_messages),
            "total_spans": int(total_spans),
            "avg_spans_per_message": float(total_spans / len(df)) if len(df) > 0 else 0.0,
            "span_type_distribution": span_type_counts,
            "unique_quoted_speakers": len(all_speakers),
            "quote_percentage": float(messages_with_quotes / len(df) * 100) if len(df) > 0 else 0.0,
        }