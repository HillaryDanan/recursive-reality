#!/usr/bin/env python3
"""
Systematic script to find recent scientific facts with citations from abstracts.
Searches Google Scholar, extracts facts from abstracts, formats for JSON.

Usage:
    python3 experiments/exp2_attribution/scripts/find_recent_facts.py

Author: Hillary Danan
Date: 2025-09-29
"""

import json
import time
from typing import List, Dict
from datetime import datetime

# This script provides a TEMPLATE for systematic searching
# You'll manually search and extract from abstracts

def create_search_template() -> List[Dict]:
    """
    Creates template with search queries for each category.
    Returns list of searches to perform manually.
    """
    
    searches = [
        # PHYSICS (6 facts needed)
        {
            "category": "physics",
            "search_query": "quantum entanglement 2020-2024 site:arxiv.org OR site:nature.com",
            "expected_facts": 2,
            "notes": "Look for clear quantitative findings in abstracts"
        },
        {
            "category": "physics", 
            "search_query": "gravitational waves detection 2020-2024 site:arxiv.org",
            "expected_facts": 2,
            "notes": "LIGO/Virgo recent discoveries"
        },
        {
            "category": "physics",
            "search_query": "superconductivity room temperature 2020-2024",
            "expected_facts": 2,
            "notes": "Recent materials science breakthroughs"
        },
        
        # BIOLOGY (6 facts needed)
        {
            "category": "biology",
            "search_query": "CRISPR gene editing 2020-2024 site:nature.com OR site:cell.com",
            "expected_facts": 2,
            "notes": "Recent CRISPR applications/discoveries"
        },
        {
            "category": "biology",
            "search_query": "microbiome human health 2020-2024 site:nature.com",
            "expected_facts": 2,
            "notes": "Gut-brain axis or immune system findings"
        },
        {
            "category": "biology",
            "search_query": "protein folding AlphaFold 2020-2024",
            "expected_facts": 2,
            "notes": "AI + biology breakthrough papers"
        },
        
        # CHEMISTRY (6 facts needed)
        {
            "category": "chemistry",
            "search_query": "catalyst design 2020-2024 site:nature.com OR site:science.org",
            "expected_facts": 2,
            "notes": "New catalytic materials/methods"
        },
        {
            "category": "chemistry",
            "search_query": "battery technology lithium 2020-2024",
            "expected_facts": 2,
            "notes": "Energy storage improvements"
        },
        {
            "category": "chemistry",
            "search_query": "carbon capture 2020-2024 site:nature.com",
            "expected_facts": 2,
            "notes": "Climate-relevant chemistry"
        },
        
        # MATHEMATICS (6 facts needed)
        {
            "category": "mathematics",
            "search_query": "prime number theorem 2020-2024 site:arxiv.org",
            "expected_facts": 2,
            "notes": "Number theory advances"
        },
        {
            "category": "mathematics",
            "search_query": "topology 2020-2024 breakthrough site:arxiv.org",
            "expected_facts": 2,
            "notes": "Recent topology proofs"
        },
        {
            "category": "mathematics",
            "search_query": "machine learning theory optimization 2020-2024",
            "expected_facts": 2,
            "notes": "Mathematical ML theory"
        },
        
        # COMPUTER SCIENCE (6 facts needed)
        {
            "category": "computer_science",
            "search_query": "transformer architecture attention 2020-2024 site:arxiv.org",
            "expected_facts": 2,
            "notes": "Neural network architecture papers"
        },
        {
            "category": "computer_science",
            "search_query": "quantum computing algorithm 2020-2024 site:arxiv.org OR site:nature.com",
            "expected_facts": 2,
            "notes": "Quantum algorithm advances"
        },
        {
            "category": "computer_science",
            "search_query": "cryptography post-quantum 2020-2024",
            "expected_facts": 2,
            "notes": "Security advances"
        }
    ]
    
    return searches


def create_fact_template(
    fact_id: int,
    category: str,
    search_query: str
) -> Dict:
    """
    Creates empty template for manual filling.
    """
    return {
        "id": fact_id,
        "fact": "FILL_THIS_FROM_ABSTRACT",
        "citation_text": "FILL_THIS",
        "full_citation": "FILL_THIS",
        "category": category,
        "complexity": "ASSESS_THIS",
        "citation_format": "ASSESS_THIS",
        "author_count": "COUNT_THIS",
        "publication_year": "FILL_THIS",
        "source_type": "journal",
        "temporal_category": "recent",
        "verification_status": "NEEDS_VERIFICATION",
        "search_query_used": search_query,
        "abstract_url": "PASTE_URL_HERE",
        "notes": "Fill from abstract - if fact not clear in abstract, skip paper"
    }


def generate_search_workflow():
    """
    Generates the complete workflow for manual searching.
    """
    print("=" * 70)
    print("SYSTEMATIC FACT FINDING WORKFLOW")
    print("=" * 70)
    print()
    
    searches = create_search_template()
    templates = []
    fact_id = 21  # Starting after the 20 classics
    
    for search in searches:
        print(f"\n{'='*70}")
        print(f"CATEGORY: {search['category'].upper()}")
        print(f"{'='*70}")
        print(f"Search query: {search['search_query']}")
        print(f"Expected facts: {search['expected_facts']}")
        print(f"Notes: {search['notes']}")
        print()
        print("STEPS:")
        print("1. Google Scholar search with query above")
        print("2. Open first 3-5 papers")
        print("3. Read ONLY the abstract")
        print("4. If main finding is clear in abstract → use it")
        print("5. If abstract is vague → skip paper (bad communication)")
        print("6. Extract: fact, citation, authors, year")
        print("7. Paste abstract URL for verification")
        print()
        
        for i in range(search['expected_facts']):
            template = create_fact_template(
                fact_id=fact_id,
                category=search['category'],
                search_query=search['search_query']
            )
            templates.append(template)
            fact_id += 1
            
            print(f"   Fact {fact_id-1}: [ ] Found")
    
    print(f"\n{'='*70}")
    print("VERIFICATION CHECKLIST")
    print(f"{'='*70}")
    print("For each fact:")
    print("[ ] Main finding stated clearly in abstract")
    print("[ ] Citation complete (authors, year, journal)")
    print("[ ] Fact paraphrased (not copy-paste from abstract)")
    print("[ ] Category assigned correctly")
    print("[ ] Complexity assessed (simple/moderate/complex)")
    print("[ ] Abstract URL saved for verification")
    print()
    
    # Save templates to file
    output_file = "experiments/exp2_attribution/data/recent_facts_TEMPLATE.json"
    
    output_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "total_needed": 30,
            "categories": {cat['category']: cat['expected_facts'] 
                          for cat in searches},
            "search_method": "google_scholar_abstract_only",
            "instructions": "Fill templates by searching, reading abstracts only, extracting clear findings"
        },
        "search_queries": searches,
        "fact_templates": templates
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Templates saved to: {output_file}")
    print(f"📋 Now: Systematically search and fill each template")
    print()


if __name__ == "__main__":
    generate_search_workflow()
