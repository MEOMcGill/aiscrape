"""Ask Google AI Mode one prompt from the desktop scraper and print what it cited.

Needs a saved Google session: `python -m aiscrape.auth --platform google`.
Google CAPTCHAs a fresh automated browser, so this will come back blocked without one.
"""

from aiscrape import scrape_ai_overview

result = scrape_ai_overview("how does photosynthesis work")

print(result.overview_text)
print()
for ref in result.references:
    print(f"  {ref.domain:24} {ref.title}")
    print(f"  {'':24} {ref.url}")
