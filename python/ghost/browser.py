"""ghost.browser — navigateur Playwright optionnel et limité aux domaines autorisés.

Le navigateur expose des observations et une navigation par liens uniquement. Il
n'exécute pas de scripts fournis par le modèle, ne soumet pas de formulaire, ne
utilise pas de session authentifiée et bloque les requêtes hors allowlist.
"""
from __future__ import annotations

from dataclasses import dataclass
import urllib.parse

from .web import WebAccessError


@dataclass(frozen=True)
class BrowserObservation:
    url: str
    title: str
    text: str
    links: tuple[tuple[str, str], ...]
    screenshot: bytes | None = None


class BrowserTool:
    def __init__(self, web_reader, *, headless=True, timeout_ms=15000,
                 viewport=(1280, 800), max_text_chars=30000, screenshot_quality=60):
        if web_reader is None:
            raise ValueError("BrowserTool exige un WebReader avec allowlist explicite.")
        self.reader = web_reader
        self.headless = bool(headless)
        self.timeout_ms = max(1000, int(timeout_ms))
        self.viewport = tuple(map(int, viewport))
        if len(self.viewport) != 2 or min(self.viewport) < 320:
            raise ValueError("viewport doit contenir une largeur et hauteur >= 320.")
        self.max_text_chars = max(1000, int(max_text_chars))
        self.screenshot_quality = max(20, min(90, int(screenshot_quality)))
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def _authorize(self, url):
        safe = self.reader._validate_url(url)
        if self.reader.respect_robots:
            self.reader._robots_allowed(safe)
        return safe

    def _guard_request(self, route):
        url = route.request.url
        scheme = urllib.parse.urlsplit(url).scheme.lower()
        if scheme in {"data", "blob", "about"}:
            route.continue_()
            return
        try:
            self.reader._validate_url(url)
            route.continue_()
        except Exception:
            route.abort()

    def _start(self):
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Le navigateur nécessite : pip install 'ghost-gsociety[browser]' puis playwright install chromium") from exc
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._context = self._browser.new_context(
                viewport={"width": self.viewport[0], "height": self.viewport[1]},
                accept_downloads=False,
                service_workers="block",
                java_script_enabled=True,
            )
            self._context.set_default_timeout(self.timeout_ms)
            self._page = self._context.new_page()
            self._page.route("**/*", self._guard_request)
        except Exception:
            self.close()
            raise

    def _observe(self, with_screenshot=False):
        if self._page is None:
            raise RuntimeError("Aucune page ouverte.")
        try:
            title = self._page.title()
        except Exception:
            title = ""
        try:
            text = self._page.locator("body").inner_text(timeout=self.timeout_ms)
        except Exception:
            text = ""
        text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
        text = text[:self.max_text_chars]
        links = []
        try:
            anchors = self._page.locator("a[href]").evaluate_all(
                "els => els.slice(0, 300).map(a => [a.href, (a.innerText || a.getAttribute('aria-label') || '').trim()])"
            )
            for href, label in anchors:
                try:
                    safe = self.reader._validate_url(href, resolve_dns=False)
                    host = urllib.parse.urlsplit(safe).hostname
                    if self.reader._is_allowed(host):
                        links.append((safe, str(label)[:300]))
                except Exception:
                    continue
        except Exception:
            pass
        screenshot = None
        if with_screenshot:
            # JPEG réduit : une capture n'est pas injectée automatiquement dans le LLM.
            screenshot = self._page.screenshot(type="jpeg", quality=self.screenshot_quality,
                                               full_page=False, animations="disabled")
        return BrowserObservation(
            url=self._page.url, title=title, text=text,
            links=tuple(links), screenshot=screenshot,
        )

    def open(self, url, *, with_screenshot=False):
        safe = self._authorize(url)
        self._start()
        response = self._page.goto(safe, wait_until="domcontentloaded", timeout=self.timeout_ms)
        if response is not None and response.status >= 400:
            raise WebAccessError(f"Navigation refusée par le serveur : HTTP {response.status}.")
        final_url = self._authorize(self._page.url)
        if final_url != self._page.url:
            raise WebAccessError("URL finale non canonique ou hors politique.")
        return self._observe(with_screenshot=with_screenshot)

    def follow_link(self, label, *, with_screenshot=False):
        """Navigue vers un lien exact par son libellé ; n'effectue aucune action de formulaire."""
        if self._page is None:
            raise RuntimeError("Ouvre une page avant de suivre un lien.")
        wanted = str(label).strip()
        if not wanted:
            raise ValueError("Le libellé du lien ne peut pas être vide.")
        links = self._page.get_by_role("link", name=wanted, exact=True)
        if links.count() < 1:
            raise WebAccessError("Aucun lien exact avec ce libellé sur la page courante.")
        href = links.first.get_attribute("href")
        if not href:
            raise WebAccessError("Lien sans destination.")
        target = self._authorize(urllib.parse.urljoin(self._page.url, href))
        return self.open(target, with_screenshot=with_screenshot)

    def close(self):
        for obj in (self._context, self._browser):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._page = self._context = self._browser = self._playwright = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
