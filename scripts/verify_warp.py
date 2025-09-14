#!/usr/bin/env python3
"""
WARP.md conformance verification script.
Confirms presence of required headings, validates config keys, and verifies
pipeline stages match spec's canonical order.
"""

import re
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any


class WarpVerifier:
    """Verify WARP.md conformance with spec requirements."""
    
    def __init__(self, warp_path: Path = Path("WARP.md"), spec_path: Path = Path("docs/spec.md")):
        self.warp_path = warp_path
        self.spec_path = spec_path
        self.errors = []
        self.warnings = []
        
    def check_required_headings(self) -> bool:
        """Check that all required headings are present."""
        required_headings = [
            "Purpose",
            "Scope", 
            "Processing Stages",
            "Quote Handling & Attribution",
            "Contextual Stance Classification",
            "Topics — Hybrid Approach", 
            "Aggregation Outputs",
            "Evaluation",
            "Milestones",
            "Risks & Mitigations"
        ]
        
        if not self.warp_path.exists():
            self.errors.append(f"WARP.md not found at {self.warp_path}")
            return False
            
        with open(self.warp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract all headings from WARP.md
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        headings_set = set(heading.strip().rstrip('—') for heading in headings)
        
        missing_headings = []
        for required in required_headings:
            # Check for exact match or close variants
            matches = [h for h in headings_set if required.lower() in h.lower() or h.lower() in required.lower()]
            if not matches:
                missing_headings.append(required)
        
        if missing_headings:
            self.errors.append(f"Missing required headings: {missing_headings}")
            return False
            
        print(f"✅ All {len(required_headings)} required headings present")
        return True
        
    def check_pipeline_stages(self) -> bool:
        """Verify 12-stage pipeline matches spec order."""
        expected_stages = [
            "3.1 Load & Normalize",
            "3.2 Language Detection", 
            "3.3 NER",
            "3.4 Entity Aliasing",
            "3.5 Quote/Span Tagging",
            "3.6 Stance Classification", 
            "3.7 Topic Analysis",
            "3.8 Sentiment & Toxicity",
            "3.9 Links & Domains",
            "3.10 Style Features",
            "3.11 Aggregations",
            "3.12 Output Writers"
        ]
        
        with open(self.warp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find Processing Stages section
        stages_match = re.search(r'## Processing Stages.*?(?=^## |\Z)', content, re.MULTILINE | re.DOTALL)
        if not stages_match:
            self.errors.append("Processing Stages section not found")
            return False
        
        stages_content = stages_match.group(0)
        
        # Extract stage headings
        found_stages = re.findall(r'### (3\.\d+\s+[^#\n]+)', stages_content)
        found_stages = [stage.strip() for stage in found_stages]
        
        if len(found_stages) != len(expected_stages):
            self.errors.append(f"Expected {len(expected_stages)} pipeline stages, found {len(found_stages)}")
            return False
            
        for i, (expected, found) in enumerate(zip(expected_stages, found_stages)):
            if expected.strip() != found.strip():
                self.errors.append(f"Stage {i+1} mismatch: expected '{expected}', found '{found}'")
                return False
        
        print(f"✅ All {len(expected_stages)} pipeline stages present in correct order")
        return True
    
    def check_config_keys(self) -> bool:
        """Validate configuration examples contain required keys."""
        required_keys = {
            'io', 'models', 'processing', 'resources', 'topic'
        }
        
        required_subkeys = {
            'models': {'ner', 'sentiment', 'toxicity', 'stance'},
            'processing': {'batch_size', 'prefer_gpu', 'quote_aware'},
            'resources': {'aliases_path', 'topics_path'}
        }
        
        with open(self.warp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find YAML config blocks
        yaml_blocks = re.findall(r'```ya?ml\n(.*?)\n```', content, re.DOTALL)
        
        if not yaml_blocks:
            self.warnings.append("No YAML configuration blocks found")
            return True
        
        # Check the main config block (usually the first substantial one)
        main_config = None
        for block in yaml_blocks:
            try:
                config = yaml.safe_load(block)
                if isinstance(config, dict) and len(config) >= 3:
                    main_config = config
                    break
            except yaml.YAMLError:
                continue
        
        if not main_config:
            self.warnings.append("No valid main configuration block found")
            return True
        
        # Check top-level keys
        missing_keys = required_keys - set(main_config.keys())
        if missing_keys:
            self.warnings.append(f"Missing config sections: {missing_keys}")
        
        # Check required subkeys
        for section, subkeys in required_subkeys.items():
            if section in main_config and isinstance(main_config[section], dict):
                missing_subkeys = subkeys - set(main_config[section].keys())
                if missing_subkeys:
                    self.warnings.append(f"Missing {section} keys: {missing_subkeys}")
        
        print(f"✅ Configuration structure validated")
        return True
        
    def check_spec_compliance(self) -> bool:
        """Run additional spec compliance checks."""
        with open(self.warp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required spec elements
        spec_elements = [
            r'neutral.*reproducible.*analytics.*pipeline',  # Purpose statement
            r'attribution.*accuracy.*transparency.*config-driven',  # Key principles
            r'quote.*detection.*speaker.*attribution',  # Quote handling
            r'hybrid.*stance.*classification',  # Hybrid stance
            r'ontology.*discovery.*topics',  # Hybrid topics
            r'evidence.*spans',  # Evidence retention
        ]
        
        found_elements = 0
        for pattern in spec_elements:
            if re.search(pattern, content, re.IGNORECASE):
                found_elements += 1
        
        compliance_ratio = found_elements / len(spec_elements)
        if compliance_ratio < 0.8:
            self.warnings.append(f"Low spec compliance: {compliance_ratio:.1%} of key elements found")
        else:
            print(f"✅ Spec compliance: {compliance_ratio:.1%} of key elements found")
        
        return True
        
    def verify(self) -> bool:
        """Run all verification checks."""
        print("🔍 Verifying WARP.md conformance...")
        
        checks = [
            self.check_required_headings,
            self.check_pipeline_stages, 
            self.check_config_keys,
            self.check_spec_compliance
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                self.errors.append(f"Check {check.__name__} failed: {e}")
                all_passed = False
        
        # Print summary
        print("\n" + "="*60)
        print("📋 WARP.MD VERIFICATION SUMMARY")
        print("="*60)
        
        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if all_passed and not self.errors:
            print("✅ All conformance checks passed!")
            return True
        else:
            print("❌ Conformance verification failed")
            return False


def main():
    parser = argparse.ArgumentParser(description="Verify WARP.md conformance")
    parser.add_argument('--warp', type=Path, default=Path('WARP.md'),
                       help='Path to WARP.md file')
    parser.add_argument('--spec', type=Path, default=Path('docs/spec.md'),
                       help='Path to spec.md file')
    parser.add_argument('--fail-on-warnings', action='store_true',
                       help='Exit with error code if warnings found')
    
    args = parser.parse_args()
    
    verifier = WarpVerifier(args.warp, args.spec)
    success = verifier.verify()
    
    # Exit with appropriate code for CI
    if not success:
        return 1
    elif args.fail_on_warnings and verifier.warnings:
        return 1
    else:
        return 0


if __name__ == '__main__':
    exit(main())