"""Send one message to Claude's web UI and print the reply.

Needs a saved session: `python -m aiscrape.auth --platform claude`. Swap in
`send_to_chatgpt` / `send_to_gemini` / `send_to_meta` for the other products --
`SEND_FNS` maps a platform name to its send function.
"""

import asyncio

from aiscrape import (
    PLATFORM_URLS,
    new_camoufox,
    send_to_claude,
    start_fresh_conversation,
    warm_chatbot_from_firefox,
)


async def main() -> None:
    async with new_camoufox(headless=True) as browser:
        ctx = await browser.new_context(storage_state="auth/claude.json")
        # Carry the real Firefox cookies across so the session looks used, not fresh.
        await warm_chatbot_from_firefox(ctx, "claude")
        page = await ctx.new_page()
        await start_fresh_conversation(page, PLATFORM_URLS["claude"])
        print(await send_to_claude(page, "Say hello in exactly five words."))


asyncio.run(main())
