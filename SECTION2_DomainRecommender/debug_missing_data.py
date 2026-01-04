#!/usr/bin/env python3
"""Debug script to test robust extraction of missing stats."""
import sys
sys.path.insert(0, '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/code')

from fetch_streamer_data import create_driver
import time

print('Debugging Missing Stats Extraction...')
driver = create_driver(headless=False)

try:
    print('Navigating to asmongold...')
    driver.get("https://twitchtracker.com/asmongold")
    time.sleep(8)
    
    # Dump body text first to see if Rank word exists
    text = driver.execute_script("return document.body.innerText")
    print("\n=== PAGE TEXT (First 1000 chars) ===")
    print(text[:1000])
    
    js_script = r"""
        const res = {};
        res.rank_spot = document.querySelector('#rank-spot') ? document.querySelector('#rank-spot').innerText : 'NULL';
        res.rank_value = document.querySelector('.rank-value') ? document.querySelector('.rank-value').innerText : 'NULL';
        
        // Search for "Rank" text
        const all = document.body.innerText;
        const rankMatch = all.match(/Rank\s*#?([0-9,]+)/i);
        res.rank_regex = rankMatch ? rankMatch[1] : 'NULL';
        
        return res;
    """
    
    results = driver.execute_script(js_script)
    print("\n=== RANK DEBUG RESULTS ===")
    print(results)

finally:
    driver.quit()
