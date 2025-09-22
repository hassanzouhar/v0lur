#!/usr/bin/env python3
"""
Charlie Kirk Deduplication Strategies

This script implements multiple approaches to reduce overrepresentation of Charlie Kirk
in the dataset while maintaining analytical value.
"""

import pandas as pd
import re
import hashlib
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime, timedelta


class CharlieKirkDeduplicator:
    """Implements various deduplication strategies for Charlie Kirk content."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.ck_mask = df['text'].str.contains('Charlie Kirk', case=False, na=False)
        self.ck_df = df[self.ck_mask].copy()
        self.non_ck_df = df[~self.ck_mask].copy()
        
        print(f"Dataset: {len(df):,} total messages")
        print(f"Charlie Kirk messages: {len(self.ck_df):,} ({len(self.ck_df)/len(df)*100:.1f}%)")
    
    def strategy_1_remove_twitter_reposts(self, keep_ratio: float = 0.1) -> pd.DataFrame:
        """
        Strategy 1: Dramatically reduce Twitter reposts
        
        Most CK content appears to be Twitter reposts. Keep only a small sample.
        """
        print(f"\n=== Strategy 1: Reduce Twitter Reposts ===")
        
        # Identify Twitter reposts
        twitter_pattern = r'Charlie Kirk \(Twitter\)|@realCharlieKirk|Charlie Kirk\s*\n'
        twitter_mask = self.ck_df['text'].str.contains(twitter_pattern, case=False, na=False)
        
        twitter_reposts = self.ck_df[twitter_mask]
        other_ck = self.ck_df[~twitter_mask]
        
        print(f"Twitter reposts: {len(twitter_reposts):,}")
        print(f"Other CK content: {len(other_ck):,}")
        
        # Keep only a small sample of Twitter reposts (e.g., 10%)
        sample_size = max(1, int(len(twitter_reposts) * keep_ratio))
        
        # Sample strategically - take every nth message to preserve time distribution
        if len(twitter_reposts) > 0:
            step = max(1, len(twitter_reposts) // sample_size)
            sampled_twitter = twitter_reposts.iloc[::step].head(sample_size)
        else:
            sampled_twitter = twitter_reposts
        
        # Combine sampled Twitter posts with other CK content
        dedupe_ck = pd.concat([sampled_twitter, other_ck])
        result_df = pd.concat([dedupe_ck, self.non_ck_df])
        
        print(f"Kept {len(sampled_twitter):,} Twitter reposts ({keep_ratio*100:.1f}% sample)")
        print(f"Final dataset: {len(result_df):,} messages")
        print(f"CK reduction: {len(self.ck_df)} → {len(dedupe_ck)} ({len(dedupe_ck)/len(self.ck_df)*100:.1f}%)")
        
        return result_df.reset_index(drop=True)
    
    def strategy_2_temporal_deduplication(self, time_window_hours: int = 24, max_per_window: int = 5) -> pd.DataFrame:
        """
        Strategy 2: Temporal deduplication
        
        Limit CK mentions to max N per time window to prevent spam-like repetition.
        """
        print(f"\n=== Strategy 2: Temporal Deduplication ===")
        print(f"Max {max_per_window} CK messages per {time_window_hours}-hour window")
        
        # Sort by date
        ck_sorted = self.ck_df.sort_values('date').copy()
        ck_sorted['date'] = pd.to_datetime(ck_sorted['date'])
        
        # Group into time windows and sample
        time_delta = pd.Timedelta(hours=time_window_hours)
        
        kept_indices = []
        current_window_start = None
        current_window_count = 0
        
        for idx, row in ck_sorted.iterrows():
            msg_time = row['date']
            
            # Check if we're in a new time window
            if (current_window_start is None or 
                msg_time >= current_window_start + time_delta):
                current_window_start = msg_time
                current_window_count = 0
            
            # Add message if under limit for this window
            if current_window_count < max_per_window:
                kept_indices.append(idx)
                current_window_count += 1
        
        dedupe_ck = ck_sorted.loc[kept_indices]
        result_df = pd.concat([dedupe_ck, self.non_ck_df])
        
        print(f"Kept {len(dedupe_ck):,} CK messages after temporal deduplication")
        print(f"CK reduction: {len(self.ck_df)} → {len(dedupe_ck)} ({len(dedupe_ck)/len(self.ck_df)*100:.1f}%)")
        
        return result_df.reset_index(drop=True)
    
    def strategy_3_content_similarity_clustering(self, similarity_threshold: float = 0.8) -> pd.DataFrame:
        """
        Strategy 3: Content similarity clustering
        
        Group similar CK messages and keep only one representative from each cluster.
        """
        print(f"\n=== Strategy 3: Content Similarity Clustering ===")
        print(f"Similarity threshold: {similarity_threshold}")
        
        def get_text_signature(text):
            """Create normalized text signature for comparison."""
            # Remove URLs, mentions, timestamps, extra whitespace
            cleaned = re.sub(r'http\S+|@\w+|\d{1,2}:\d{2}|\d{1,2}/\d{1,2}', '', str(text))
            cleaned = re.sub(r'\s+', ' ', cleaned.lower().strip())
            # Remove common boilerplate
            cleaned = re.sub(r'charlie kirk \(twitter\)', '', cleaned)
            return cleaned
        
        # Create signatures for all CK messages
        signatures = self.ck_df['text'].apply(get_text_signature)
        
        # Find similarity clusters
        clusters = []
        processed = set()
        
        for i, sig1 in enumerate(signatures):
            if i in processed:
                continue
                
            cluster = [i]
            for j, sig2 in enumerate(signatures):
                if i != j and j not in processed:
                    similarity = SequenceMatcher(None, sig1, sig2).ratio()
                    if similarity >= similarity_threshold:
                        cluster.append(j)
            
            clusters.append(cluster)
            processed.update(cluster)
        
        # Keep one representative from each cluster (prefer earlier messages)
        kept_indices = []
        cluster_sizes = []
        
        for cluster in clusters:
            # Sort cluster by date and keep the earliest
            cluster_msgs = self.ck_df.iloc[cluster].sort_values('date')
            kept_indices.append(cluster_msgs.index[0])
            cluster_sizes.append(len(cluster))
        
        dedupe_ck = self.ck_df.loc[kept_indices]
        result_df = pd.concat([dedupe_ck, self.non_ck_df])
        
        print(f"Found {len(clusters):,} similarity clusters")
        print(f"Average cluster size: {np.mean(cluster_sizes):.1f}")
        print(f"Largest cluster: {max(cluster_sizes)} messages")
        print(f"Kept {len(dedupe_ck):,} representative messages")
        print(f"CK reduction: {len(self.ck_df)} → {len(dedupe_ck)} ({len(dedupe_ck)/len(self.ck_df)*100:.1f}%)")
        
        return result_df.reset_index(drop=True)
    
    def strategy_4_content_type_balancing(self) -> pd.DataFrame:
        """
        Strategy 4: Content type balancing
        
        Maintain diversity by limiting each type of CK content to reasonable proportions.
        """
        print(f"\n=== Strategy 4: Content Type Balancing ===")
        
        # Define content categories and limits
        categories = {
            'twitter_repost': (r'Charlie Kirk \(Twitter\)', 100),  # Limit Twitter reposts
            'show_promo': (r'Charlie Kirk Show', 50),              # Limit show promotions  
            'retweets': (r'Retweeted by Charlie Kirk', 30),        # Limit retweets
            'podcast': (r'(podcast|episode)', 30),                 # Limit podcast content
            'tpusa': (r'(Turning Point|TPUSA)', 40),              # Limit TPUSA content
        }
        
        categorized = {'other': []}
        
        # Categorize messages
        for category, (pattern, limit) in categories.items():
            mask = self.ck_df['text'].str.contains(pattern, case=False, na=False)
            categorized[category] = self.ck_df[mask].index.tolist()
        
        # Find uncategorized messages
        all_categorized = set()
        for cat_indices in categorized.values():
            all_categorized.update(cat_indices)
        
        categorized['other'] = [idx for idx in self.ck_df.index if idx not in all_categorized]
        
        # Sample from each category
        kept_indices = []
        
        for category, indices in categorized.items():
            if category == 'other':
                # Keep all uncategorized content
                kept_indices.extend(indices)
                limit_used = len(indices)
            else:
                pattern, limit = categories[category]
                # Sample up to limit, preserving temporal distribution
                if len(indices) <= limit:
                    kept_indices.extend(indices)
                    limit_used = len(indices)
                else:
                    # Take every nth message to preserve time distribution
                    step = len(indices) // limit
                    sampled = indices[::step][:limit]
                    kept_indices.extend(sampled)
                    limit_used = len(sampled)
            
            print(f"{category}: {len(indices):,} → {limit_used:,}")
        
        dedupe_ck = self.ck_df.loc[kept_indices]
        result_df = pd.concat([dedupe_ck, self.non_ck_df])
        
        print(f"CK reduction: {len(self.ck_df)} → {len(dedupe_ck)} ({len(dedupe_ck)/len(self.ck_df)*100:.1f}%)")
        
        return result_df.reset_index(drop=True)
    
    def strategy_5_hybrid_approach(self) -> pd.DataFrame:
        """
        Strategy 5: Hybrid approach combining multiple methods
        
        1. Remove most Twitter reposts
        2. Apply temporal limits
        3. Remove near-duplicates
        """
        print(f"\n=== Strategy 5: Hybrid Approach ===")
        
        # Step 1: Reduce Twitter reposts (keep 15%)
        twitter_pattern = r'Charlie Kirk \(Twitter\)'
        twitter_mask = self.ck_df['text'].str.contains(twitter_pattern, case=False, na=False)
        
        twitter_reposts = self.ck_df[twitter_mask]
        other_ck = self.ck_df[~twitter_mask]
        
        # Sample Twitter reposts
        sample_size = max(10, int(len(twitter_reposts) * 0.15))
        step = max(1, len(twitter_reposts) // sample_size)
        sampled_twitter = twitter_reposts.iloc[::step].head(sample_size)
        
        # Combine with other CK content
        step1_ck = pd.concat([sampled_twitter, other_ck])
        print(f"Step 1 - Twitter reduction: {len(self.ck_df)} → {len(step1_ck)}")
        
        # Step 2: Temporal limits (max 3 per 12-hour window)
        step1_sorted = step1_ck.sort_values('date').copy()
        step1_sorted['date'] = pd.to_datetime(step1_sorted['date'])
        
        time_delta = pd.Timedelta(hours=12)
        max_per_window = 3
        
        kept_indices = []
        current_window_start = None
        current_window_count = 0
        
        for idx, row in step1_sorted.iterrows():
            msg_time = row['date']
            
            if (current_window_start is None or 
                msg_time >= current_window_start + time_delta):
                current_window_start = msg_time
                current_window_count = 0
            
            if current_window_count < max_per_window:
                kept_indices.append(idx)
                current_window_count += 1
        
        step2_ck = step1_sorted.loc[kept_indices]
        print(f"Step 2 - Temporal limits: {len(step1_ck)} → {len(step2_ck)}")
        
        # Step 3: Remove near-duplicates (quick hash-based approach)
        def quick_hash(text):
            # Simple hash of first 100 chars for fast dedup
            clean_text = re.sub(r'\s+', ' ', str(text).lower()[:100])
            return hashlib.md5(clean_text.encode()).hexdigest()[:8]
        
        step2_ck['text_hash'] = step2_ck['text'].apply(quick_hash)
        step3_ck = step2_ck.drop_duplicates(subset=['text_hash'], keep='first').drop('text_hash', axis=1)
        
        print(f"Step 3 - Remove duplicates: {len(step2_ck)} → {len(step3_ck)}")
        
        # Final result
        result_df = pd.concat([step3_ck, self.non_ck_df])
        
        print(f"Final CK reduction: {len(self.ck_df)} → {len(step3_ck)} ({len(step3_ck)/len(self.ck_df)*100:.1f}%)")
        print(f"Final dataset: {len(result_df):,} messages")
        
        return result_df.reset_index(drop=True)
    
    def analyze_impact(self, original_df: pd.DataFrame, dedupe_df: pd.DataFrame) -> Dict:
        """Analyze the impact of deduplication on entity distributions."""
        
        def get_entity_stats(df):
            all_entities = []
            for entities_list in df['entities']:
                if entities_list:
                    all_entities.extend([e['text'] for e in entities_list])
            return Counter(all_entities)
        
        orig_entities = get_entity_stats(original_df)
        dedupe_entities = get_entity_stats(dedupe_df)
        
        print(f"\n=== Impact Analysis ===")
        print(f"Dataset size: {len(original_df):,} → {len(dedupe_df):,}")
        print(f"Total entities: {sum(orig_entities.values()):,} → {sum(dedupe_entities.values()):,}")
        
        print("\nTop 10 entities before/after:")
        for entity, orig_count in orig_entities.most_common(10):
            new_count = dedupe_entities.get(entity, 0)
            change = ((new_count - orig_count) / orig_count * 100) if orig_count > 0 else 0
            print(f"{entity}: {orig_count:,} → {new_count:,} ({change:+.1f}%)")
        
        return {
            'original_size': len(original_df),
            'dedupe_size': len(dedupe_df),
            'reduction_ratio': len(dedupe_df) / len(original_df),
            'original_entities': orig_entities,
            'dedupe_entities': dedupe_entities
        }


def main():
    """Demonstrate all deduplication strategies."""
    
    # Load data
    df = pd.read_parquet('out/run-20250917-0818/posts_enriched_utf8_clean.parquet')
    
    deduplicator = CharlieKirkDeduplicator(df)
    
    # Run all strategies and save results
    strategies = [
        ('twitter_reduction', lambda d: d.strategy_1_remove_twitter_reposts(keep_ratio=0.1)),
        ('temporal_dedupe', lambda d: d.strategy_2_temporal_deduplication(time_window_hours=24, max_per_window=5)),
        ('content_clustering', lambda d: d.strategy_3_content_similarity_clustering(similarity_threshold=0.8)),
        ('content_balancing', lambda d: d.strategy_4_content_type_balancing()),
        ('hybrid_approach', lambda d: d.strategy_5_hybrid_approach()),
    ]
    
    results = {}
    
    for name, strategy_func in strategies:
        print(f"\n{'='*60}")
        print(f"RUNNING: {name.upper()}")
        print(f"{'='*60}")
        
        dedupe_df = strategy_func(deduplicator)
        
        # Save result
        output_path = f'out/run-20250917-0818/posts_enriched_dedupe_{name}.parquet'
        dedupe_df.to_parquet(output_path, engine='fastparquet')
        print(f"Saved: {output_path}")
        
        # Analyze impact
        impact = deduplicator.analyze_impact(df, dedupe_df)
        results[name] = impact
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Strategy':<20} {'Messages':<12} {'Reduction':<12} {'CK Messages':<12}")
    print("-" * 60)
    
    original_ck = len(deduplicator.ck_df)
    
    for name, impact in results.items():
        size = impact['dedupe_size']
        reduction = f"{impact['reduction_ratio']*100:.1f}%"
        ck_after = impact['dedupe_entities'].get('Charlie Kirk', 0)
        
        print(f"{name:<20} {size:<12,} {reduction:<12} {ck_after:<12,}")
    
    print(f"{'Original':<20} {len(df):<12,} {'100.0%':<12} {original_ck:<12,}")


if __name__ == "__main__":
    main()