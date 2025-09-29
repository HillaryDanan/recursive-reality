#!/usr/bin/env python3
"""
Intelligent batch research tool that uses Claude's web_search 
to find 30 recent papers efficiently.

This is a RUNNER script - copy the search instructions and 
paste into a NEW Claude chat to leverage web_search capabilities.

Author: Hillary Danan
Date: 2025-09-29
"""

import json
from datetime import datetime


def generate_research_batch():
    """
    Generates comprehensive search instructions for Claude to execute.
    """
    
    research_task = """
# RESEARCH TASK: Find 30 Recent Scientific Papers (2020-2024)

Use web_search to find papers with CLEAR findings stated in abstracts.

## DISTRIBUTION NEEDED:
- Physics: 6 papers
- Biology: 6 papers
- Chemistry: 6 papers  
- Mathematics: 6 papers
- Computer Science: 6 papers

## SEARCH STRATEGY:
Use these EXACT search queries with web_search:

### Physics (need 6):
1. gravitational waves detection 2020-2024 site:nature.com OR site:arxiv.org
2. room temperature superconductivity 2022-2024 site:nature.com
3. quantum computing breakthrough 2020-2024 site:arxiv.org

### Biology (need 6):
4. AlphaFold protein structure 2020-2024 site:nature.com
5. mRNA vaccine technology 2020-2023 site:science.org OR site:nature.com
6. microbiome gut-brain axis 2020-2024 site:nature.com

### Chemistry (need 6):
7. carbon capture technology 2022-2024 site:nature.com OR site:science.org
8. green hydrogen production 2020-2024 site:nature.com
9. MOF metal organic frameworks 2020-2024 site:nature.com

### Mathematics (need 6):
10. machine learning optimization theory 2020-2024 site:arxiv.org
11. graph neural networks theory 2020-2024 site:arxiv.org  
12. differential privacy algorithms 2020-2024 site:arxiv.org

### Computer Science (need 6):
13. large language model scaling laws 2020-2024 site:arxiv.org
14. federated learning privacy 2020-2024 site:arxiv.org
15. quantum algorithms complexity 2020-2024 site:arxiv.org OR site:nature.com

## FOR EACH PAPER FOUND:

Extract from abstract ONLY and format as JSON:

{
  "id": [21-50],
  "fact": "According to [Authors] ([Year]), [MAIN FINDING from abstract in YOUR OWN WORDS].",
  "citation_text": "[First Author] et al. ([Year])" or "[Author] ([Year])",
  "full_citation": "[Full citation from paper]",
  "category": "physics|biology|chemistry|mathematics|computer_science",
  "complexity": "simple|moderate|complex",
  "citation_format": "narrative",
  "author_count": [1-10],
  "publication_year": [2020-2024],
  "source_type": "journal",
  "temporal_category": "recent",
  "verification_status": "VERIFY_THIS",
  "abstract_url": "[Direct URL to paper/abstract]",
  "notes": "[One sentence about finding]"
}

## CRITICAL RULES:
1. ONLY use facts clearly stated in abstracts
2. If abstract is vague → skip that paper, find another
3. Paraphrase facts in YOUR words (no copy-paste)
4. Get at least 2 papers per search query
5. Verify publication year is 2020-2024
6. Include direct URL to abstract

## OUTPUT FORMAT:
Return valid JSON array with all 30 papers, ready to save to file.
"""
    
    print(research_task)
    return research_task


def save_research_template():
    """Save template for manual completion if needed."""
    
    template = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "total_needed": 30,
            "method": "claude_web_search_automation",
            "instructions": "This file will be populated by Claude's web_search"
        },
        "papers": []
    }
    
    with open('experiments/exp2_attribution/data/recent_papers_AUTO.json', 'w') as f:
        json.dump(template, f, indent=2)
    
    print("✅ Template saved: experiments/exp2_attribution/data/recent_papers_AUTO.json")


if __name__ == "__main__":
    print("="*80)
    print("AUTOMATED RESEARCH INSTRUCTIONS")
    print("="*80)
    print()
    print("COPY THE TEXT BELOW and paste into a NEW Claude chat")
    print("Claude will use web_search to find all 30 papers automatically")
    print()
    print("="*80)
    print()
    
    generate_research_batch()
    
    print()
    print("="*80)
    print("After Claude completes the research, it will return JSON")
    print("Save that JSON to: experiments/exp2_attribution/data/recent_papers_AUTO.json")
    print("="*80)
    
    save_research_template()