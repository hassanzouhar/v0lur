#!/usr/bin/env python3
"""
Demo script for the Telegram Analysis Textual UI

Shows a preview of what the UI will display using your actual data.
"""

from data_loaders import discover_runs, load_daily_summary, load_topic_analysis, load_entity_counts
from formatting import (
    fmt_int, fmt_float, fmt_pct, fmt_date,
    sentiment_color, toxicity_color, confidence_color, count_color,
    format_badge_text, format_run_timestamp
)

def main():
    print("🎯 TELEGRAM ANALYSIS TEXTUAL UI DEMO")
    print("=" * 60)
    
    # Discover runs
    runs = discover_runs("out")
    print(f"\n📂 DISCOVERED RUNS ({len(runs)} total):")
    
    for i, run in enumerate(runs, 1):
        badge = format_badge_text(run.available_files)
        timestamp = format_run_timestamp(run.timestamp)
        print(f"  {i}. {run.name} [{badge}] - {timestamp}")
    
    if not runs:
        print("  No runs found. Run your analysis first!")
        return
    
    # Show detailed data for the first run
    best_run = runs[0]
    print(f"\n🔍 DETAILED VIEW: {best_run.name}")
    print("-" * 40)
    
    # Summary Panel Preview
    summary_file = best_run.get_file_path("channel_daily_summary.csv")
    if summary_file:
        print("\n📊 SUMMARY PANEL:")
        data = load_daily_summary(summary_file)
        if data:
            totals = data['totals']
            sentiment_val = totals['avg_sentiment'] 
            toxicity_val = totals['avg_toxicity']
            
            print(f"  📈 Total Messages: {fmt_int(totals['total_messages'])}")
            print(f"  😊 Avg Sentiment: {fmt_float(sentiment_val, 3)} ({sentiment_color(sentiment_val)})")
            print(f"  ☠️  Avg Toxicity: {fmt_float(toxicity_val, 3)} ({toxicity_color(toxicity_val)})")
            print(f"  📅 Date Range: {totals['date_range']}")
            print(f"  🗺️ Days Covered: {fmt_int(totals['num_days'])}")
    
    # Topics Panel Preview
    topics_file = best_run.get_file_path("channel_topic_analysis.json")
    if topics_file:
        print("\n🏷️  TOPICS PANEL:")
        data = load_topic_analysis(topics_file)
        if data:
            metadata = data['metadata']
            print(f"  🏷️ Topics Found: {fmt_int(metadata['unique_topics'])}")
            print(f"  🎯 Avg Confidence: {fmt_float(metadata['avg_confidence'], 3)}")
            print(f"  ⭐ High Confidence: {fmt_pct(metadata['confident_percentage'])}")
            
            print("  Top Topics:")
            for i, topic in enumerate(data['topics'][:3], 1):
                confidence_col = confidence_color(topic['confidence'])
                print(f"    {i}. {topic['topic']}: {fmt_int(topic['count'])} ({fmt_pct(topic['percentage'])}) - {confidence_col}")
    
    # Entities Panel Preview  
    entities_file = best_run.get_file_path("channel_entity_counts.csv")
    if entities_file:
        print("\n👥 ENTITIES PANEL:")
        data = load_entity_counts(entities_file)
        if data:
            metadata = data['metadata']
            print(f"  👥 Unique Entities: {fmt_int(metadata['total_entities'])}")
            print(f"  📊 Total Mentions: {fmt_int(metadata['total_mentions'])}")
            
            print("  Top Entities:")
            for i, entity in enumerate(data['entities'][:3], 1):
                print(f"    {i}. {entity['entity']}: {fmt_int(entity['count'])} ({fmt_pct(entity['percentage'])})")
    
    print("\n" + "=" * 60)
    print("🚀 READY TO LAUNCH FULL UI!")
    print("Run: .venv/bin/python3.11 textual_ui.py")
    print("=" * 60)

if __name__ == "__main__":
    main()