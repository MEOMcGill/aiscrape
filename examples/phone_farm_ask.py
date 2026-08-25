"""Ask Google AI Mode and ChatGPT on one Android handset, over adb + CDP.

Both scrapers share a single `PhoneChromeSession`: the CDP tunnel binds a port
that a second session would collide with. Find your serial with `adb devices`.
"""

from aiscrape import PhoneChatGPTScraper, PhoneChromeSession, PhoneFarmAIOverviewScraper

SERIAL = "R58MEXAMPLE"       # adb devices
SSH_HOST = None              # or "phone-farm" when the handsets hang off another box

with PhoneChromeSession(SERIAL, ssh_host=SSH_HOST) as phone:
    google = PhoneFarmAIOverviewScraper(session=phone)
    chatgpt = PhoneChatGPTScraper(session=phone)

    ai = google.search("how does photosynthesis work")
    print(f"AI Mode ({ai.surface}): {ai.overview_text[:200]}")
    for ref in ai.references:
        print(f"  cited: {ref.domain}")

    gpt = chatgpt.ask("how does photosynthesis work")
    print(f"\nChatGPT ({gpt.served_model or 'anonymous'}): {gpt.response[:200]}")
    for ref in gpt.references:
        print(f"  cited: {ref.domain}")
