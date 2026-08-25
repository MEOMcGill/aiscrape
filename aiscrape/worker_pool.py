"""Pool of Workers consuming tasks from a shared asyncio.Queue.

One WorkerPool drives one provider: it spins up to `max_workers` Workers (capped
by how many active accounts that provider's pool has), each holding its own
account + camoufox session, and load-balances submitted Tasks across them.
Ported from igscrape's WorkerPool.
"""

import asyncio

from .accounts_pool import AccountsPool
from .exceptions import NoAccountError
from .logger import logger
from .models import Task
from .worker import QUERIES_PER_REST, Worker


class WorkerPool:
    def __init__(
        self,
        pool: AccountsPool,
        max_workers: int = 3,
        queries_per_rest: int = QUERIES_PER_REST,
        headless: bool = True,
        settle_ms: int = 9_000,
        refresh_from_firefox: bool = True,
    ):
        self.pool = pool
        self.provider = pool.provider
        self.max_workers = max_workers
        self.queries_per_rest = queries_per_rest
        self.headless = headless
        self.settle_ms = settle_ms
        self.refresh_from_firefox = refresh_from_firefox

        self.workers: list[Worker] = []
        self.worker_tasks: list[asyncio.Task] = []
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self._initialized = False
        self._shutdown = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> int:
        if self._initialized:
            return len(self.workers)

        active = await self.pool.get_active_accounts()
        if not active:
            raise NoAccountError(f"No active {self.provider} accounts in pool")

        num = max(1, min(self.max_workers, len(active)))
        logger.info(
            f"WorkerPool[{self.provider}] initializing {num} workers "
            f"(max={self.max_workers}, active={len(active)})"
        )

        for i in range(num):
            try:
                worker = await Worker.create(
                    id=f"{self.provider}-worker-{i}",
                    pool=self.pool,
                    provider=self.provider,
                    queries_per_rest=self.queries_per_rest,
                    headless=self.headless,
                    settle_ms=self.settle_ms,
                    refresh_from_firefox=self.refresh_from_firefox,
                )
                self.workers.append(worker)
                self.worker_tasks.append(asyncio.create_task(self._worker_loop(worker)))
            except NoAccountError:
                logger.warning(
                    f"WorkerPool[{self.provider}]: only created "
                    f"{len(self.workers)}/{num} workers"
                )
                break

        if not self.workers:
            raise NoAccountError(f"Failed to create any {self.provider} workers")

        self._initialized = True
        return len(self.workers)

    async def _worker_loop(self, worker: Worker):
        logger.info(f"{worker.id} loop started")
        while not self._shutdown:
            try:
                task, future = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            logger.info(f"{worker.id} processing {task.endpoint}")
            try:
                result = await worker.execute_task(task)
                future.set_result(result)
            except Exception as e:
                logger.error(f"{worker.id} task failed: {e}")
                future.set_exception(e)
            finally:
                self.task_queue.task_done()

        logger.info(f"{worker.id} loop exiting")

    async def submit_task(self, task: Task) -> asyncio.Future:
        async with self._init_lock:
            if not self._initialized:
                await self.initialize()

        future = asyncio.get_running_loop().create_future()
        await self.task_queue.put((task, future))
        return future

    async def submit(self, task: Task):
        """Submit a task and await its result."""
        return await (await self.submit_task(task))

    async def close(self):
        if not self._initialized:
            return
        self._shutdown = True
        for t in self.worker_tasks:
            t.cancel()
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        for worker in self.workers:
            await worker.close()
        self.workers = []
        self.worker_tasks = []
        self._initialized = False
        self._shutdown = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
