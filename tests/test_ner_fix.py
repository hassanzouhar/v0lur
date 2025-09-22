#!/usr/bin/env python3
"""Test script to verify NER entity type mapping fixes."""

import sys
import os
sys.path.insert(0, 'src')

from raigem0n.processors.ner_processor import NERProcessor
from raigem0n.config import Config

def test_ner_fixes():
    """Test that the NER processor correctly handles entity type mapping and filtering."""
    
    # Load configuration
    config = Config("config/config.yaml")
    aliases = config.load_aliases()
    
    # Initialize NER processor
    ner_processor = NERProcessor(
        model_name="dslim/bert-base-NER",
        max_entities_per_msg=10,  # Allow more for testing
        device="cpu"  # Force CPU for testing
    )
    
    # Load aliases
    ner_processor.load_aliases(aliases)
    
    # Test texts that should trigger our fixes
    test_texts = [
        "The Democrats and Twitter are discussing the issue.",
        "Republicans support Trump's policies on Twitter.",
        "DNC officials met with Democratic Party leaders.",
        "The United States and China are negotiating.",
        "Charlie Kirk spoke about iden and Ka issues.",  # Should filter garbage entities
        "Show me the anti-establishment Twitter posts."  # Should filter garbage words
    ]
    
    print("Testing NER processor with fixes...")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        print("-" * 30)
        
        entities = ner_processor.extract_entities([text])[0]
        
        if not entities:
            print("No entities found")
            continue
            
        for entity in entities:
            print(f"Entity: '{entity['text']}' | Type: {entity['type']} | Confidence: {entity['confidence']:.3f}")
            if entity.get('original_text') and entity['original_text'] != entity['text']:
                print(f"  (Originally: '{entity['original_text']}')")
    
    print("\n" + "=" * 50)
    print("Key fixes to verify:")
    print("1. 'Democrats' -> 'Democratic Party' with type ORG")
    print("2. 'DNC' -> 'Democratic Party' with type ORG") 
    print("3. 'Republicans' -> 'Republican Party' with type ORG")
    print("4. 'Twitter' should have type ORG")
    print("5. 'United States' should have type LOC")
    print("6. Garbage entities like 'iden', 'Ka', 'Show', 'anti' should be filtered out")

if __name__ == "__main__":
    test_ner_fixes()