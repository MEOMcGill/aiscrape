"""Interactive DOM probe: study the real AI Overview structure to pick selectors.

Paths resolve against the current working directory.

Usage:
  python -m aiscrape.probe "how does photosynthesis work"
"""
import asyncio, json, sys
from pathlib import Path
from urllib.parse import quote_plus

from aiscrape.browser import new_camoufox

AUTH = Path("auth/google.json")

JS_PROBE = r"""
() => {
  const out = {};
  // 1. Find the "AI Overview" label node (smallest element whose trimmed text == label)
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
  let labelEl = null;
  while (walker.nextNode()) {
    const el = walker.currentNode;
    const t = (el.textContent || '').trim();
    if ((t === 'AI Overview' || t === 'AI overview') && el.children.length <= 2) {
      labelEl = el; break;
    }
  }
  const desc = (el) => el ? {
    tag: el.tagName, id: el.id || null,
    cls: (el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className) || null,
    attrs: Object.fromEntries([...el.attributes].map(a => [a.name, a.value])),
    textLen: (el.innerText || '').length,
  } : null;
  out.label = desc(labelEl);

  // 2. Ancestor chain of the label (up to 12 levels) with text length at each
  const chain = [];
  let cur = labelEl;
  for (let i = 0; cur && i < 12; i++) {
    chain.push({ lvl: i, ...desc(cur) });
    cur = cur.parentElement;
  }
  out.ancestorChain = chain;

  // 3. "Show more" buttons anywhere
  out.showMore = [...document.querySelectorAll('div[role=button],button,a,span[role=button]')]
    .filter(e => /show more|show all|more\b/i.test((e.innerText||'').trim()) && (e.innerText||'').length < 40)
    .slice(0,8).map(e => ({txt:(e.innerText||'').trim().slice(0,30), tag:e.tagName,
       aria:e.getAttribute('aria-expanded'), cls:(typeof e.className==='string'?e.className:'').slice(0,80)}));

  // 4. Pick the best container: walk up from label until text length jumps big, capture data-* attrs
  let container = labelEl;
  while (container && (container.innerText||'').length < 200 && container.parentElement) {
    container = container.parentElement;
  }
  out.container = desc(container);
  // data-attrs present on container subtree (sample of unique data-* attribute names)
  if (container) {
    const names = new Set();
    container.querySelectorAll('*').forEach(e => [...e.attributes].forEach(a => {
      if (a.name.startsWith('data-')) names.add(a.name);
    }));
    out.dataAttrsInContainer = [...names].slice(0, 40);
    // 5. Links inside container: href + text
    out.links = [...container.querySelectorAll('a[href]')].slice(0, 50).map(a => ({
      href: a.getAttribute('href').slice(0, 120),
      text: (a.innerText||'').trim().slice(0, 60),
      aria: (a.getAttribute('aria-label')||'').slice(0,60),
    }));
  }
  return out;
}
"""

async def main(q):
    storage = str(AUTH) if AUTH.exists() else None
    async with new_camoufox(headless=True, locale="en-CA", geoip=True, os=["windows","macos"]) as b:
        ctx = await b.new_context(storage_state=storage)
        page = await ctx.new_page()
        await page.goto(f"https://www.google.com/search?q={quote_plus(q)}&hl=en&gl=ca", timeout=60_000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(7000)
        # expand show more before probing links
        for sel in ('div[role="button"]:has-text("Show more")','button:has-text("Show more")'):
            try:
                btn = page.locator(sel).first
                if await btn.count() and await btn.is_visible():
                    await btn.click(timeout=3000); await page.wait_for_timeout(2000); break
            except Exception: pass
        res = await page.evaluate(JS_PROBE)
        print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv)>1 else "how does photosynthesis work"))
