#!/usr/bin/env python3
"""
Automated recent paper finder using Claude's web search.
This script is a RUNNER that calls Claude to do the research.

The actual research happens in a separate Claude session using web_search.
This script just formats the results.

Usage:
    python3 experiments/exp2_attribution/scripts/auto_find_recent_papers.py

Author: Hillary Danan  
Date: 2025-09-29
"""

import json
from datetime import datetime

def create_research_instructions():
    """
    Creates instructions for Claude to research papers.
    """
    
    instructions = """
RESEARCH TASK: Find 30 recent scientific papers (2020-2024) with clear findings in abstracts.

REQUIREMENTS:
- Use web_search to find papers
- Read abstracts only
- Extract main finding, citation, authors, year
- Skip papers with vague abstracts
- Ensure peer-reviewed sources (Nature, Science, Cell, arXiv, etc.)

DISTRIBUTION:
- Physics: 6 papers
- Biology: 6 papers  
- Chemistry: 6 papers
- Mathematics: 6 papers
- Computer Science: 6 papers

SEARCH QUERIES (use these with web_search):
1. "quantum entanglement 2020-2024 site:nature.com OR site:arxiv.org"
2. "CRISPR breakthrough 2020-2024 site:nature.com"
3. "battery technology 2020-2024 site:science.org"
4. "transformer architecture 2020-2024 site:arxiv.org"
... (continue for 30 papers)

OUTPUT FORMAT for each paper:
{
  "id": 21,
  "fact": "According to [Authors] ([Year]), [main finding from abstract].",
  "citation_text": "[Authors] ([Year])",
  "full_citation": "[Authors]. ([Year]). [Title]. [Journal], [Vol](Issue), [Pages].",
  "category": "physics|biology|chemistry|mathematics|computer_science",
  "complexity": "simple|moderate|complex",
  "citation_format": "narrative|parenthetical",
  "author_count": [number],
  "publication_year": [YYYY],
  "source_type": "journal",
  "temporal_category": "recent",
  "verification_status": "VERIFY_THIS",
  "abstract_url": "[URL]",
  "notes": "[Brief note about finding]"
}

CRITICAL: Only use facts clearly stated in abstracts. If abstract is vague, skip the paper.
"""
    
    print(instructions)
    return instructions


if __name__ == "__main__":
    print("="*80)
    print("AUTOMATED PAPER FINDER - INSTRUCTIONS")
    print("="*80)
    print()
    print("This script provides instructions for Claude to research papers.")
    print("Copy the instructions below and paste into a NEW Claude chat.")
    print()
    create_research_instructions()
