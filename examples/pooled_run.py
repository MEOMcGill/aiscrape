"""Fan a batch of prompts across several pooled Google sessions.

Seed the pool first: `python -m aiscrape.auth --platform google`, then
`aiscrape --provider google seed-from-auth`. The pool rotates an account when it
trips a rate-limit and marks it inactive when its session expires.
"""

import asyncio

from aiscrape import AccountsPool, Task, WorkerPool

PROMPTS = [
    "how does photosynthesis work",
    "what is a black hole",
    "why is the sky blue",
]


async def main() -> None:
    pool = AccountsPool("google")                       # db/google.db
    async with WorkerPool(pool, max_workers=3) as wp:
        results = await asyncio.gather(
            *[wp.submit(Task("ai_mode", {"prompt": p})) for p in PROMPTS]
        )
        for prompt, result in zip(PROMPTS, results):
            print(f"{prompt}: {result.overview_text[:100]}")


asyncio.run(main())
