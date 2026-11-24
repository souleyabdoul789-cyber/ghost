"""ghost.web — lecture contrôlée de pages Web publiques et autorisées.

La couche refuse par défaut les domaines non explicitement autorisés,
respecte robots.txt, limite la taille des réponses et bloque les adresses
IP locales/privées pour éviter les usages SSRF.
"""

from dataclasses import dataclass
from html.parser import HTMLParser
import hashlib
import ipaddress
import socket
import time
import urllib.error
import urllib.parse
import urllib.robotparser
import urllib.request


class WebAccessError(RuntimeError):
    """La requête Web a été refusée ou n'a pas pu être lue proprement."""


@dataclass(frozen=True)
class WebDocument:
    url: str
    title: str
    text: str
    links: tuple
    fetched_at: float
    sha256: str
    content_type: str


class _PageParser(HTMLParser):
    BLOCK_TAGS = {
        "article", "section", "p", "div", "main", "h1", "h2", "h3",
        "h4", "h5", "h6", "li", "blockquote", "pre", "br", "tr",
    }
    SKIP_TAGS = {
        "script", "style", "noscript", "svg", "canvas", "iframe",
        "template", "nav", "footer", "aside", "form", "button",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.links = []
        self.skip_depth = 0
        self.title_depth = 0
        self.title_parts = []
        self.anchor_depth = 0
        self.anchor_href = None
        self.anchor_parts = []
        self.meta_description = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.title_depth += 1
        if tag == "meta" and attrs.get("name", "").lower() == "description":
            self.meta_description = attrs.get("content", "").strip()
        if tag == "a" and attrs.get("href"):
            self.anchor_depth += 1
            self.anchor_href = attrs["href"]
            self.anchor_parts = []
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        if tag == "a" and self.anchor_depth:
            label = " ".join(" ".join(self.anchor_parts).split())
            if self.anchor_href:
                self.links.append((self.anchor_href, label))
            self.anchor_depth = 0
            self.anchor_href = None
            self.anchor_parts = []
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self.title_depth:
            self.title_parts.append(text)
        self.parts.append(text + " ")
        if self.anchor_depth:
            self.anchor_parts.append(text)

    def result(self):
        text = "\n".join(line.strip() for line in "".join(self.parts).splitlines() if line.strip())
        title = " ".join(self.title_parts).strip()
        if not text and self.meta_description:
            text = self.meta_description
        return title, text, tuple(self.links)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, validate_url):
        super().__init__()
        self.validate_url = validate_url

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        if new.scheme != "https" or not new.hostname:
            raise WebAccessError("Redirection refusée : destination hors liste autorisée ou non HTTPS.")
        if old.scheme == "https" and new.scheme != "https":
            raise WebAccessError("Une redirection HTTPS vers HTTP est interdite.")
        # La destination est validée (domaine + DNS public) AVANT de suivre
        # la redirection, pas seulement après téléchargement.
        self.validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WebReader:
    """Télécharge et extrait du HTML public sur une liste de domaines explicite.

    Exemple :
        reader = WebReader(allowed_domains=["docs.python.org"])
        document = reader.fetch("https://docs.python.org/3/library/index.html")

    Seules les URLs HTTPS sont acceptées. robots.txt est respecté par défaut.
    """

    def __init__(self, allowed_domains, user_agent="G-SOCIETY-ResearchBot/0.1",
                 timeout=12, max_bytes=2_000_000, respect_robots=True,
                 min_interval=0.5):
        if isinstance(allowed_domains, str):
            allowed_domains = [allowed_domains]
        domains = set()
        for item in (allowed_domains or []):
            raw = str(item).strip().strip(".")
            try:
                domain = raw.encode("idna").decode("ascii").lower()
            except UnicodeError as exc:
                raise ValueError("Nom de domaine invalide dans allowed_domains.") from exc
            domains.add(domain)
        if not domains or any(not d or "/" in d or ":" in d or " " in d for d in domains):
            raise ValueError("Fournis au moins un domaine explicite valide dans allowed_domains.")
        if float(timeout) <= 0 or int(max_bytes) <= 0 or float(min_interval) < 0:
            raise ValueError("timeout et max_bytes doivent être positifs ; min_interval non négatif.")
        self.allowed_domains = frozenset(domains)
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self.max_bytes = int(max_bytes)
        self.respect_robots = bool(respect_robots)
        self.min_interval = max(0.0, float(min_interval))
        self._robots = {}
        self._last_request = {}
        self._opener = urllib.request.build_opener(_SafeRedirectHandler(self._validate_url))

    def _is_allowed(self, host):
        host = (host or "").lower().rstrip(".")
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)

    def _validate_url(self, url, resolve_dns=True):
        try:
            parsed = urllib.parse.urlsplit(str(url))
            port = parsed.port or 443
        except ValueError as exc:
            raise WebAccessError("URL mal formée.") from exc
        if parsed.scheme != "https" or not parsed.hostname:
            raise WebAccessError("Seules les URLs HTTPS publiques sont acceptées.")
        if parsed.username or parsed.password:
            raise WebAccessError("Les URLs avec identifiants intégrés sont interdites.")
        if ":" in parsed.hostname:
            raise WebAccessError("Les adresses IPv6 littérales ne sont pas acceptées.")
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        if not self._is_allowed(host):
            raise WebAccessError("Domaine absent de la liste autorisée : " + host)
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            raise WebAccessError("Les noms d'hôtes locaux sont interdits.")
        if resolve_dns:
            try:
                addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except OSError as exc:
                raise WebAccessError("Résolution DNS impossible pour " + host) from exc
            for item in addresses:
                try:
                    address = ipaddress.ip_address(item[4][0].split("%", 1)[0])
                except ValueError as exc:
                    raise WebAccessError("Adresse DNS invalide retournée pour " + host) from exc
                if not address.is_global:
                    raise WebAccessError("Adresse IP non publique refusée : " + str(address))
        return urllib.parse.urlunsplit(("https", host + ((":" + str(port)) if port != 443 else ""),
                                        parsed.path or "/", parsed.query, ""))

    def _request(self, url, accept="text/html,application/xhtml+xml,text/plain;q=0.8"):
        parsed = urllib.parse.urlsplit(url)
        origin = parsed.netloc
        wait = self.min_interval - (time.monotonic() - self._last_request.get(origin, 0.0))
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept": accept,
            "Accept-Encoding": "identity",
        })
        self._last_request[origin] = time.monotonic()
        try:
            response = self._opener.open(request, timeout=self.timeout)
            with response:
                final_url = self._validate_url(response.geturl())
                content_type = response.headers.get_content_type().lower()
                raw = response.read(self.max_bytes + 1)
                if len(raw) > self.max_bytes:
                    raise WebAccessError("Réponse trop volumineuse : limite max_bytes dépassée.")
                charset = response.headers.get_content_charset() or "utf-8"
                return final_url, content_type, charset, raw
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise WebAccessError("Ressource introuvable (HTTP 404).") from exc
            raise WebAccessError("Erreur HTTP " + str(exc.code)) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise WebAccessError("Échec de la requête HTTPS : " + str(exc)) from exc

    def _robots_allowed(self, url):
        parsed = urllib.parse.urlsplit(url)
        origin = parsed.scheme + "://" + parsed.netloc
        if origin not in self._robots:
            robots_url = origin + "/robots.txt"
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                _, content_type, charset, raw = self._request(robots_url, "text/plain")
                parser.parse(raw.decode(charset, errors="replace").splitlines())
            except WebAccessError as exc:
                # HTTP 404 => aucune politique publiée. Les autres erreurs ferment l'accès.
                if "404" in str(exc):
                    parser.parse([])
                else:
                    raise WebAccessError("Impossible de vérifier robots.txt : " + str(exc)) from exc
            self._robots[origin] = parser
        parser = self._robots[origin]
        if not parser.can_fetch(self.user_agent, url):
            raise WebAccessError("Accès interdit par robots.txt pour cette URL.")
        return parser.crawl_delay(self.user_agent) or parser.crawl_delay("*") or 0.0

    def fetch(self, url):
        safe_url = self._validate_url(url)
        delay = self._robots_allowed(safe_url) if self.respect_robots else 0.0
        if delay > self.min_interval:
            time.sleep(min(float(delay), 30.0))
        final_url, content_type, charset, raw = self._request(safe_url)
        if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
            raise WebAccessError("Type de contenu non pris en charge : " + content_type)
        decoded = raw.decode(charset, errors="replace")
        if content_type == "text/plain":
            title, text, links = "", decoded.strip(), ()
        else:
            parser = _PageParser()
            parser.feed(decoded)
            title, text, raw_links = parser.result()
            links = []
            for href, label in raw_links:
                target = urllib.parse.urljoin(final_url, href)
                try:
                    target = self._validate_url(target, resolve_dns=False)
                except (ValueError, WebAccessError):
                    continue
                links.append((target, label))
            links = tuple(links)
        digest = hashlib.sha256(raw).hexdigest()
        return WebDocument(
            url=final_url, title=title, text=text, links=links,
            fetched_at=time.time(), sha256=digest, content_type=content_type,
        )
