"""PhoneBackend's handling of the ChatGPT memory setting — no handset."""

import asyncio

from aiscrape.phone_chatgpt import ChatGPTMemoryError
from aiscrape.phone_pool import PhoneBackend, SerialPool


class _Chat:
    def __init__(self, error=None):
        self.error = error

    def apply_memory(self):
        if self.error:
            raise self.error
        return False


def _memory_verdict(chat) -> bool | None:
    async def run():
        pool = SerialPool(["S1"])
        backend = PhoneBackend(pool, chatgpt_memory=False)
        backend._chatgpt = chat
        await backend._apply_chatgpt_memory("S1")
        return await pool.chatgpt_state("S1")
    return asyncio.run(run())


def test_memory_that_cannot_be_set_takes_the_handset_out_of_chat():
    assert _memory_verdict(_Chat(ChatGPTMemoryError("S1: memory is on, wanted off"))) is False


def test_memory_set_leaves_the_handset_unproven():
    assert _memory_verdict(_Chat()) is None
