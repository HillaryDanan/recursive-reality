#!/usr/bin/env python3
"""
Interactive citation verification tool.
Shows papers one by one, you type Y/N, it handles the rest.

Usage:
    python3 experiments/exp2_attribution/scripts/verify_citations.py

Controls:
    Y = Verified (mark as good)
    N = Skip/Bad (mark as rejected)  
    S = Save for later (keep as needs verification)
    Q = Quit (saves progress)

Author: Hillary Danan
Date: 2025-09-29
"""

import json
import os
import webbrowser
from typing import Dict, List

# ANSI color codes for pretty terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
END = '\033[0m'


def load_facts(filepath: str) -> Dict:
    """Load facts JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_facts(filepath: str, data: Dict):
    """Save facts JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"{GREEN}✅ Progress saved{END}")


def get_google_scholar_url(citation: str, year: int) -> str:
    """Generate Google Scholar search URL."""
    # Extract author and year for search
    search_query = citation.replace(',', '').replace('(', '').replace(')', '')
    search_query = search_query.strip() + f" {year}"
    encoded = search_query.replace(' ', '+')
    return f"https://scholar.google.com/scholar?q={encoded}"


def display_fact(fact: Dict, current: int, total: int):
    """Display a single fact with formatting."""
    print("\n" + "="*80)
    print(f"{BOLD}{BLUE}FACT {current}/{total}{END}")
    print("="*80)
    
    # Basic info
    print(f"{BOLD}ID:{END} {fact['id']}")
    print(f"{BOLD}Category:{END} {fact['category']}")
    print(f"{BOLD}Year:{END} {fact['publication_year']}")
    print(f"{BOLD}Status:{END} {fact['verification_status']}")
    print()
    
    # The fact itself
    print(f"{BOLD}FACT:{END}")
    print(f"  {fact['fact']}")
    print()
    
    # Citation
    print(f"{BOLD}CITATION:{END}")
    print(f"  {fact['citation_text']}")
    print()
    print(f"{BOLD}FULL CITATION:{END}")
    print(f"  {fact['full_citation']}")
    print()
    
    # Google Scholar link
    scholar_url = get_google_scholar_url(fact['citation_text'], fact['publication_year'])
    print(f"{BOLD}GOOGLE SCHOLAR SEARCH:{END}")
    print(f"  {scholar_url}")
    print()
    
    # Notes if any
    if fact.get('notes'):
        print(f"{BOLD}NOTES:{END} {fact['notes']}")
        print()


def verify_facts_interactive(filepath: str):
    """Main interactive verification loop."""
    
    # Load data
    data = load_facts(filepath)
    facts = data['facts']
    
    # Count verification status
    verified = sum(1 for f in facts if f['verification_status'] == 'VERIFIED')
    needs_verify = sum(1 for f in facts if f['verification_status'] == 'VERIFY_THIS')
    rejected = sum(1 for f in facts if f['verification_status'] == 'REJECTED')
    
    print(f"\n{BOLD}CITATION VERIFICATION TOOL{END}")
    print(f"Total facts: {len(facts)}")
    print(f"{GREEN}Verified: {verified}{END}")
    print(f"{YELLOW}Needs verification: {needs_verify}{END}")
    print(f"{RED}Rejected: {rejected}{END}")
    print()
    print(f"{BOLD}Controls:{END}")
    print(f"  {GREEN}Y{END} = Verified (mark as good)")
    print(f"  {RED}N{END} = Rejected (mark as bad)")
    print(f"  {YELLOW}S{END} = Skip (save for later)")
    print(f"  {BLUE}Q{END} = Quit (saves progress)")
    print(f"  {BLUE}O{END} = Open URL in browser")
    print()
    
    try:
        current = 0
        for idx, fact in enumerate(facts):
            # Only show facts that need verification
            if fact['verification_status'] != 'VERIFY_THIS':
                continue
            
            current += 1
            display_fact(fact, current, needs_verify)
            
            # Get user input
            while True:
                choice = input(f"{BOLD}Verify? (Y/N/S/O/Q): {END}").strip().upper()
                
                if choice == 'Y':
                    fact['verification_status'] = 'VERIFIED'
                    print(f"{GREEN}✅ Marked as VERIFIED{END}")
                    save_facts(filepath, data)
                    break
                    
                elif choice == 'N':
                    fact['verification_status'] = 'REJECTED'
                    reason = input(f"  Reason for rejection (optional): ").strip()
                    if reason:
                        fact['rejection_reason'] = reason
                    print(f"{RED}❌ Marked as REJECTED{END}")
                    save_facts(filepath, data)
                    break
                    
                elif choice == 'S':
                    print(f"{YELLOW}⏭️  Skipped - will show again next time{END}")
                    break
                    
                elif choice == 'O':
                    scholar_url = get_google_scholar_url(fact['citation_text'], fact['publication_year'])
                    print(f"Opening: {scholar_url}")
                    try:
                        webbrowser.open(scholar_url)
                        print(f"{GREEN}✅ Opened in browser{END}")
                    except:
                        print(f"{RED}❌ Couldn't open browser - copy URL above{END}")
                    # Don't break - let them verify after opening
                    
                elif choice == 'Q':
                    print(f"\n{YELLOW}Quitting and saving progress...{END}")
                    save_facts(filepath, data)
                    print(f"{GREEN}✅ Progress saved - run again to continue{END}")
                    return
                    
                else:
                    print(f"{RED}Invalid choice. Use Y/N/S/O/Q{END}")
        
        # Done with all facts
        print("\n" + "="*80)
        print(f"{GREEN}{BOLD}🎉 VERIFICATION COMPLETE!{END}")
        print("="*80)
        
        # Final stats
        verified = sum(1 for f in facts if f['verification_status'] == 'VERIFIED')
        rejected = sum(1 for f in facts if f['verification_status'] == 'REJECTED')
        
        print(f"{GREEN}Verified: {verified}{END}")
        print(f"{RED}Rejected: {rejected}{END}")
        print()
        
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Interrupted - saving progress...{END}")
        save_facts(filepath, data)
        print(f"{GREEN}✅ Progress saved{END}")


if __name__ == "__main__":
    filepath = "experiments/exp2_attribution/data/facts_with_citations.json"
    
    if not os.path.exists(filepath):
        print(f"{RED}Error: {filepath} not found{END}")
        print(f"Make sure you're in the repo root directory")
        exit(1)
    
    verify_facts_interactive(filepath)
