#!/usr/bin/env python3
"""Debug script to inspect page text for stats extraction."""
import sys
sys.path.insert(0, '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/code')

from fetch_streamer_data import create_driver
import time

print('Debugging DOM for stats...')
driver = create_driver(headless=False)

try:
    print('Navigating to ninja page...')
    driver.get("https://twitchtracker.com/ninja")
    time.sleep(10)
    
    # Dump body text
    text = driver.execute_script("return document.body.innerText")
    print("\n=== PAGE TEXT SAMPLE ===")
    print(text[:2000]) # First 2000 chars
    
    # Try to find "viewers"
    import re
    matches = re.findall(r'.{0,20}viewers.{0,20}', text, re.IGNORECASE)
    print("\n=== VIEWER MATCHES ===")
    for m in matches:
        print(f"Match: '{m}'")
        
    # Check stats grid specifically
    stats_text = driver.execute_script("""
        const stats = document.querySelectorAll('div, span, td');
        let found = [];
        stats.forEach(el => {
            if (el.textContent.toLowerCase().includes('viewers')) {
                found.push(el.innerText);
            }
        });
        return found.slice(0, 5);
    """)
    print("\n=== JS STATS FINDER ===")
    print(stats_text)

finally:
    driver.quit()
