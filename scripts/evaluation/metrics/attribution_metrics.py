"""
Attribution accuracy metrics for evaluating speaker attribution.
Part of Milestone 7 evaluation framework.
"""

from typing import Any, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_attribution_metrics(predictions: List[Dict], gold_standard: List[Dict]) -> Dict[str, Any]:
    """Compute speaker attribution accuracy metrics.
    
    Args:
        predictions: List of predicted span annotations
        gold_standard: List of gold standard span annotations
        
    Returns:
        Dictionary with attribution accuracy metrics
    """
    if len(predictions) != len(gold_standard):
        logger.warning(f"Mismatch in prediction/gold lengths: {len(predictions)} vs {len(gold_standard)}")
    
    # Align by message ID
    aligned_pairs = align_predictions_by_message_id(predictions, gold_standard)
    
    # Span-level metrics
    span_metrics = compute_span_level_metrics(aligned_pairs)
    
    # Message-level metrics  
    message_metrics = compute_message_level_metrics(aligned_pairs)
    
    # Quote detection metrics
    quote_metrics = compute_quote_detection_metrics(aligned_pairs)
    
    return {
        **span_metrics,
        **message_metrics,
        **quote_metrics,
        'total_messages_evaluated': len(aligned_pairs)
    }


def align_predictions_by_message_id(predictions: List[Dict], gold_standard: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Align predictions with gold standard by message ID."""
    # Create lookups by message ID
    pred_by_id = {item['message_id']: item for item in predictions}
    gold_by_id = {item['message_id']: item for item in gold_standard}
    
    # Find common message IDs
    common_ids = set(pred_by_id.keys()) & set(gold_by_id.keys())
    
    aligned_pairs = []
    for msg_id in common_ids:
        aligned_pairs.append((pred_by_id[msg_id], gold_by_id[msg_id]))
    
    logger.info(f"Aligned {len(aligned_pairs)} messages for attribution evaluation")
    
    return aligned_pairs


def compute_span_level_metrics(aligned_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, float]:
    """Compute span-level attribution accuracy metrics."""
    total_spans = 0
    correct_attributions = 0
    
    # Speaker type counts
    speaker_stats = {
        'author': {'total': 0, 'correct': 0},
        'quoted': {'total': 0, 'correct': 0},
        'forwarded': {'total': 0, 'correct': 0},
        'unknown': {'total': 0, 'correct': 0}
    }
    
    for pred_item, gold_item in aligned_pairs:
        pred_spans = pred_item.get('spans', [])
        gold_spans = gold_item.get('spans', [])
        
        # Match spans by overlap (IoU > 0.5)
        span_matches = match_spans_by_overlap(pred_spans, gold_spans)
        
        for pred_span, gold_span in span_matches:
            total_spans += 1
            
            # Check speaker attribution
            pred_speaker = pred_span.get('speaker', 'unknown')
            gold_speaker = gold_span.get('speaker', 'unknown')
            
            if pred_speaker == gold_speaker:
                correct_attributions += 1
                speaker_stats[gold_speaker]['correct'] += 1
            
            speaker_stats[gold_speaker]['total'] += 1
    
    # Compute overall accuracy
    span_accuracy = correct_attributions / total_spans if total_spans > 0 else 0.0
    
    # Compute per-speaker-type accuracy
    speaker_accuracies = {}
    for speaker_type, stats in speaker_stats.items():
        if stats['total'] > 0:
            speaker_accuracies[f'{speaker_type}_accuracy'] = stats['correct'] / stats['total']
        else:
            speaker_accuracies[f'{speaker_type}_accuracy'] = 0.0
    
    return {
        'span_accuracy': span_accuracy,
        'total_spans_evaluated': total_spans,
        'correct_attributions': correct_attributions,
        **speaker_accuracies
    }


def compute_message_level_metrics(aligned_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, float]:
    """Compute message-level attribution accuracy."""
    total_messages = len(aligned_pairs)
    perfect_messages = 0
    
    for pred_item, gold_item in aligned_pairs:
        pred_spans = pred_item.get('spans', [])
        gold_spans = gold_item.get('spans', [])
        
        # Check if all spans in message are correctly attributed
        if is_perfect_attribution(pred_spans, gold_spans):
            perfect_messages += 1
    
    message_accuracy = perfect_messages / total_messages if total_messages > 0 else 0.0
    
    return {
        'message_accuracy': message_accuracy,
        'perfect_attribution_messages': perfect_messages
    }


def compute_quote_detection_metrics(aligned_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, float]:
    """Compute quote detection precision, recall, and F1."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_item, gold_item in aligned_pairs:
        pred_spans = pred_item.get('spans', [])
        gold_spans = gold_item.get('spans', [])
        
        # Find quoted spans
        pred_quoted = [s for s in pred_spans if s.get('span_type') in ['quoted', 'forwarded']]
        gold_quoted = [s for s in gold_spans if s.get('span_type') in ['quoted', 'forwarded']]
        
        # Match quoted spans
        matched_quotes = match_spans_by_overlap(pred_quoted, gold_quoted)
        
        true_positives += len(matched_quotes)
        false_positives += len(pred_quoted) - len(matched_quotes)
        false_negatives += len(gold_quoted) - len(matched_quotes)
    
    # Compute precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'quote_detection_precision': precision,
        'quote_detection_recall': recall,
        'quote_detection_f1': f1,
        'quote_true_positives': true_positives,
        'quote_false_positives': false_positives,
        'quote_false_negatives': false_negatives
    }


def match_spans_by_overlap(pred_spans: List[Dict], gold_spans: List[Dict], iou_threshold: float = 0.5) -> List[Tuple[Dict, Dict]]:
    """Match predicted and gold spans by IoU overlap threshold."""
    matches = []
    used_gold_indices = set()
    
    for pred_span in pred_spans:
        best_match = None
        best_iou = 0.0
        best_gold_idx = -1
        
        for gold_idx, gold_span in enumerate(gold_spans):
            if gold_idx in used_gold_indices:
                continue
                
            iou = compute_span_iou(pred_span, gold_span)
            if iou > best_iou and iou >= iou_threshold:
                best_match = gold_span
                best_iou = iou
                best_gold_idx = gold_idx
        
        if best_match is not None:
            matches.append((pred_span, best_match))
            used_gold_indices.add(best_gold_idx)
    
    return matches


def compute_span_iou(span1: Dict, span2: Dict) -> float:
    """Compute Intersection over Union (IoU) for two spans."""
    start1, end1 = span1.get('start', 0), span1.get('end', 0)
    start2, end2 = span2.get('start', 0), span2.get('end', 0)
    
    # Compute intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Compute union
    union = (end1 - start1) + (end2 - start2) - intersection
    
    return intersection / union if union > 0 else 0.0


def is_perfect_attribution(pred_spans: List[Dict], gold_spans: List[Dict]) -> bool:
    """Check if all spans in a message have perfect attribution."""
    if len(pred_spans) != len(gold_spans):
        return False
    
    # Match all spans
    matches = match_spans_by_overlap(pred_spans, gold_spans, iou_threshold=0.8)
    
    if len(matches) != len(gold_spans):
        return False
    
    # Check all attributions are correct
    for pred_span, gold_span in matches:
        if pred_span.get('speaker', 'unknown') != gold_span.get('speaker', 'unknown'):
            return False
    
    return True


def analyze_attribution_errors(aligned_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, Any]:
    """Analyze types of attribution errors for debugging."""
    error_types = {
        'span_boundary_errors': 0,
        'speaker_type_errors': 0,
        'missed_quotes': 0,
        'false_quotes': 0,
        'multi_speaker_errors': 0
    }
    
    examples = {
        'boundary_errors': [],
        'type_errors': [],
        'missed_quotes': [],
        'false_quotes': []
    }
    
    for pred_item, gold_item in aligned_pairs:
        pred_spans = pred_item.get('spans', [])
        gold_spans = gold_item.get('spans', [])
        
        # Analyze span matching issues
        matches = match_spans_by_overlap(pred_spans, gold_spans, iou_threshold=0.3)
        
        # Count unmatched spans (boundary errors)
        unmatched_pred = len(pred_spans) - len(matches)
        unmatched_gold = len(gold_spans) - len(matches)
        error_types['span_boundary_errors'] += unmatched_pred + unmatched_gold
        
        # Analyze attribution errors in matched spans
        for pred_span, gold_span in matches:
            pred_type = pred_span.get('span_type', 'author')
            gold_type = gold_span.get('span_type', 'author')
            
            if pred_type != gold_type:
                error_types['speaker_type_errors'] += 1
                
                # Collect examples
                if len(examples['type_errors']) < 10:
                    examples['type_errors'].append({
                        'text': pred_span.get('text', '')[:100],
                        'predicted_type': pred_type,
                        'gold_type': gold_type
                    })
        
        # Check for missed quotes
        gold_quotes = [s for s in gold_spans if s.get('span_type') in ['quoted', 'forwarded']]
        pred_quotes = [s for s in pred_spans if s.get('span_type') in ['quoted', 'forwarded']]
        
        if len(gold_quotes) > len(pred_quotes):
            error_types['missed_quotes'] += len(gold_quotes) - len(pred_quotes)
        elif len(pred_quotes) > len(gold_quotes):
            error_types['false_quotes'] += len(pred_quotes) - len(gold_quotes)
    
    return {
        'error_counts': error_types,
        'error_examples': examples
    }