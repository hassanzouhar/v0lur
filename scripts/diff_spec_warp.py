#!/usr/bin/env python3
"""
Spec-to-WARP alignment verification script.
Checks that all required sections from docs/spec.md are present in WARP.md
with correct headings, configuration keys, and content alignment.
"""

import re
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set


def extract_headings(file_path: Path) -> List[str]:
    """Extract markdown headings from a file."""
    headings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                # Extract heading text, remove # and whitespace
                heading = re.sub(r'^#+\s*', '', line).strip()
                # Remove trailing markdown like --- or parentheses
                heading = re.sub(r'\s*---.*$', '', heading)
                heading = re.sub(r'\s*\(.*\)$', '', heading)
                headings.append(heading)
    return headings


def extract_config_keys(file_path: Path) -> Set[str]:
    """Extract YAML configuration keys from markdown code blocks."""
    config_keys = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find YAML code blocks
    yaml_blocks = re.findall(r'```ya?ml\n(.*?)\n```', content, re.DOTALL)
    
    for block in yaml_blocks:
        lines = block.split('\n')
        for line in lines:
            # Extract keys (handle nested keys with dots)
            if ':' in line and not line.strip().startswith('#'):
                key_match = re.match(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', line)
                if key_match:
                    indent, key = key_match.groups()
                    # Simple key extraction - could be enhanced for nested structures
                    config_keys.add(key)
    
    return config_keys


def extract_pipeline_stages(file_path: Path) -> List[str]:
    """Extract processing pipeline stage names."""
    stages = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for numbered stages like "3.1 Load & Normalize" or "### 3.1 Load & Normalize"
    stage_patterns = [
        r'###?\s*3\.(\d+)\s+([^#\n]+)',  # ### 3.1 Stage Name
        r'^\s*(\d+)\.\s*\*\*([^*]+)\*\*',  # 1. **Stage Name**
    ]
    
    for pattern in stage_patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        for match in matches:
            if len(match) == 2:
                stage_name = match[1].strip()
                if stage_name:
                    stages.append(stage_name)
    
    return stages


def check_section_alignment() -> List[Dict]:
    """Check alignment between spec.md sections and WARP.md."""
    
    spec_path = Path('docs/spec.md')
    warp_path = Path('WARP.md')
    
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    if not warp_path.exists():
        raise FileNotFoundError(f"WARP file not found: {warp_path}")
    
    # Extract data from both files
    spec_headings = set(extract_headings(spec_path))
    warp_headings = set(extract_headings(warp_path))
    
    spec_config_keys = extract_config_keys(spec_path)
    warp_config_keys = extract_config_keys(warp_path)
    
    spec_stages = extract_pipeline_stages(spec_path)
    warp_stages = extract_pipeline_stages(warp_path)
    
    # Define required sections from spec
    required_sections = [
        "Purpose",
        "Scope", 
        "Data Model",
        "Processing Stages",
        "Quote Handling & Attribution",
        "Contextual Stance Classification", 
        "Topics — Hybrid Approach",
        "Links & Domains",
        "Style Features",
        "Aggregation Outputs",
        "Config Example",
        "Evaluation",
        "Milestones",
        "Risks & Mitigations",
        "Deliverables",
        "Guiding Principles"
    ]
    
    # Check alignment
    results = []
    
    # Section headings check
    for section in required_sections:
        # Check for exact match or close variant
        exact_match = section in warp_headings
        close_matches = [h for h in warp_headings if section.lower() in h.lower() or h.lower() in section.lower()]
        
        exists = exact_match or len(close_matches) > 0
        action = "✅ Present" if exists else "❌ Missing"
        matches_found = close_matches[:2] if close_matches else []
        
        results.append({
            "spec_section": section,
            "required_elements": "Section heading",
            "exists_in_WARP": exists,
            "action": action,
            "details": f"Matches: {matches_found}" if matches_found else "",
            "owner": "doc-alignment"
        })
    
    # Config keys check
    missing_config_keys = spec_config_keys - warp_config_keys
    for key in missing_config_keys:
        results.append({
            "spec_section": "Configuration",
            "required_elements": f"Config key: {key}",
            "exists_in_WARP": False,
            "action": "❌ Missing config key",
            "details": f"Key '{key}' found in spec but not in WARP",
            "owner": "config-alignment"
        })
    
    # Pipeline stages check  
    if len(spec_stages) > 0 and len(warp_stages) > 0:
        stages_match = set(spec_stages) == set(warp_stages)
        results.append({
            "spec_section": "Processing Stages",
            "required_elements": "12-stage pipeline order",
            "exists_in_WARP": stages_match,
            "action": "✅ Stages aligned" if stages_match else "⚠️ Stage mismatch",
            "details": f"Spec: {len(spec_stages)} stages, WARP: {len(warp_stages)} stages",
            "owner": "pipeline-alignment"
        })
    
    return results


def generate_report(results: List[Dict], output_path: Path):
    """Generate CSV alignment report."""
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['spec_section', 'required_elements', 'exists_in_WARP', 'action', 'details', 'owner']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"📊 Alignment report written to: {output_path}")


def print_summary(results: List[Dict]):
    """Print alignment summary to console."""
    
    total = len(results)
    present = len([r for r in results if r['exists_in_WARP']])
    missing = total - present
    
    print("\n" + "="*60)
    print("📋 SPEC-WARP ALIGNMENT SUMMARY")
    print("="*60)
    print(f"Total checks:     {total}")
    print(f"✅ Present:       {present}")
    print(f"❌ Missing:       {missing}")
    print(f"📈 Coverage:      {present/total*100:.1f}%")
    
    if missing > 0:
        print(f"\n⚠️  Missing elements:")
        for result in results:
            if not result['exists_in_WARP']:
                print(f"  • {result['spec_section']}: {result['required_elements']}")
    
    print("\n✨ Use the CSV report for detailed tracking and next steps.")


def main():
    parser = argparse.ArgumentParser(description="Check WARP.md alignment with spec.md")
    parser.add_argument('--output', '-o', type=Path, 
                       default=Path('.reports/spec_warp_alignment.csv'),
                       help='Output CSV file path')
    parser.add_argument('--json', action='store_true',
                       help='Also output JSON format')
    
    args = parser.parse_args()
    
    try:
        results = check_section_alignment()
        
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate reports
        generate_report(results, args.output)
        
        if args.json:
            json_path = args.output.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"📄 JSON report written to: {json_path}")
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())