#!/usr/bin/env python3
"""Debug script to inspect GAMES page text for game extraction."""
import sys
sys.path.insert(0, '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/code')

from fetch_streamer_data import create_driver
import time

print('Debugging GAMES page...')
driver = create_driver(headless=False)

try:
    print('Navigating to ninja/games page...')
    driver.get("https://twitchtracker.com/ninja/games")
    time.sleep(8)
    
    # Dump body text
    text = driver.execute_script("return document.body.innerText")
    print("\n=== GAMES PAGE TEXT SAMPLE ===")
    print(text[:3000]) # First 3000 chars to see table headers and first few rows
    
    # Check table specifically
    table_text = driver.execute_script("""
        const table = document.querySelector('table');
        return table ? table.innerText : 'NO TABLE FOUND';
    """)
    print("\n=== TABLE TEXT ===")
    print(table_text[:2000])

finally:
    driver.quit()
