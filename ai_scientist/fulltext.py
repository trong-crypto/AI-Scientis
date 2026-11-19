"""Retrieve verified open-access XML or indexed PDF without claiming paywalled access.

DOI-matched Europe PMC XML is tried first; an indexed public PDF is a fallback.
Unextractable or mismatched works remain explicitly marked abstract-only.
"""

import re
import ipaddress
import socket
from io import BytesIO
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import requests
from pypdf import PdfReader


BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MAX_XML_BYTES = 4_000_000
MAX_PASSAGES = 120
MAX_TEXT_CHARS = 120_000
MAX_PDF_BYTES = 8_000_000
MAX_PDF_PAGES = 40
TIMEOUT = 12


def _plain(element):
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def parse_open_article_xml(xml_bytes):
    """Return section- and paragraph-addressable text from JATS body and tables."""
    if len(xml_bytes) > MAX_XML_BYTES:
        raise ValueError("Full-text XML exceeds the size limit.")
    root = ElementTree.fromstring(xml_bytes)
    body = root.find(".//body")
    if body is None:
        raise ValueError("Open article has no readable body.")
    parents = {child: parent for parent in body.iter() for child in parent}
    passages = []
    used = 0
    for element in body.iter():
        if element.tag not in ("p", "table-wrap"):
            continue
        parent = parents.get(element)
        if element.tag == "p" and any(ancestor.tag in ("table-wrap", "fig")
                                  for ancestor in _ancestors(parent, parents)):
            continue
        if element.tag == "table-wrap":
            caption = element.find("./caption")
            rows = [" | ".join(_plain(cell) for cell in row if cell.tag in ("td", "th"))
                    for row in element.findall(".//tr")]
            value = "; ".join(filter(None, [(_plain(caption) if caption is not None else ""),
                                             *rows]))
        else:
            value = _plain(element)
        if len(value) < 35:
            continue
        section = next((ancestor for ancestor in _ancestors(parent, parents)
                        if ancestor.tag == "sec"), None)
        title_node = section.find("./title") if section is not None else None
        section_title = _plain(title_node) if title_node is not None else "Body"
        text = value[:3000]
        if used + len(text) > MAX_TEXT_CHARS or len(passages) >= MAX_PASSAGES:
            break
        passages.append({"location": f"{section_title}, passage {len(passages) + 1}",
                         "section": section_title, "kind": "table" if element.tag == "table-wrap" else "paragraph",
                         "text": text})
        used += len(text)
    if not passages:
        raise ValueError("Open article body contains no usable paragraphs or tables.")
    return passages


def _ancestors(parent, parents):
    while parent is not None:
        yield parent
        parent = parents.get(parent)


def _public_https_url(url):
    parts = urlsplit(url or "")
    host = parts.hostname
    if parts.scheme != "https" or not host or parts.username or parts.password:
        return False
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        return bool(addresses) and all(ipaddress.ip_address(entry[4][0]).is_global
                                       for entry in addresses)
    except (OSError, ValueError):
        return False


def _read_open_pdf(url, session):
    """Download a bounded public HTTPS PDF, validating each redirect."""
    for _ in range(4):
        parts = urlsplit(url or "")
        if parts.scheme == "http" and not parts.username and not parts.password:
            url = urlunsplit(("https", parts.netloc, parts.path, parts.query, parts.fragment))
        if not _public_https_url(url):
            raise ValueError("Open PDF URL is not public HTTPS.")
        response = session.get(url, headers={"Accept": "application/pdf"}, timeout=TIMEOUT,
                               stream=True, allow_redirects=False)
        try:
            if getattr(response, "status_code", 200) in (301, 302, 303, 307, 308):
                location = response.headers.get("Location", "")
                if not location:
                    raise ValueError("Open PDF redirect has no target.")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                data.extend(chunk)
                if len(data) > MAX_PDF_BYTES:
                    raise ValueError("Open PDF exceeds the size limit.")
            if not data.startswith(b"%PDF-"):
                raise ValueError("Open link did not return a PDF.")
            return bytes(data), url
        finally:
            response.close()
    raise ValueError("Too many open PDF redirects.")


def parse_open_pdf(pdf_bytes, title):
    """Extract page-addressable passages, rejecting unrelated or unreadable PDFs."""
    if len(pdf_bytes) > MAX_PDF_BYTES or not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Not a bounded PDF.")
    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    if reader.is_encrypted:
        raise ValueError("Open PDF is encrypted.")
    pages = []
    for number, page in enumerate(reader.pages[:MAX_PDF_PAGES], 1):
        value = re.sub(r"[ \t]+", " ", page.extract_text() or "")
        reference_heading = re.search(r"(?im)^\s*(?:references|bibliography)\s*$", value)
        if reference_heading:
            value = value[:reference_heading.start()]
        if value.strip():
            pages.append((number, value))
        if reference_heading:
            break
    first_text = " ".join(value for _, value in pages[:2]).casefold()
    title_words = [word for word in re.findall(r"\w{4,}", title.casefold())
                   if word not in {"study", "review", "with", "from", "using", "their", "based"}]
    if len(first_text) < 500 or (len(title_words) >= 2 and
                                 sum(word in first_text for word in set(title_words[:8])) < 2):
        raise ValueError("PDF text is unreadable or its title does not match the indexed work.")
    passages = []
    used = 0
    for number, value in pages:
        for paragraph in re.split(r"\n\s*\n", value):
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if len(paragraph) < 35:
                continue
            for start in range(0, len(paragraph), 2500):
                text = paragraph[start:start + 2500]
                if used + len(text) > MAX_TEXT_CHARS or len(passages) >= MAX_PASSAGES:
                    break
                passages.append({"location": f"page {number}, passage {len(passages) + 1}",
                                 "section": f"Page {number}", "kind": "pdf_text", "text": text})
                used += len(text)
            if used >= MAX_TEXT_CHARS or len(passages) >= MAX_PASSAGES:
                break
        if used >= MAX_TEXT_CHARS or len(passages) >= MAX_PASSAGES:
            break
    if not passages:
        raise ValueError("Open PDF contains no extractable passages.")
    return passages


def fetch_open_fulltext(source, session):
    """Try DOI-matched Europe PMC XML, then an indexed open-access PDF."""
    doi = source.get("doi") or ""
    xml_error = "No DOI-matched open-access XML found."
    if doi and len(doi) <= 200:
        try:
            response = session.get(BASE + "/search", params={"query": f"DOI:{doi}",
                                   "format": "json", "pageSize": 3}, timeout=TIMEOUT)
            response.raise_for_status()
            records = response.json().get("resultList", {}).get("result", [])
            match = next((item for item in records if str(item.get("doi", "")).casefold() == doi.casefold()
                          and str(item.get("isOpenAccess", "")).upper() == "Y" and
                          re.fullmatch(r"PMC\d+", str(item.get("pmcid", "")))), None)
            if match is not None:
                url = f"{BASE}/{quote(match['pmcid'])}/fullTextXML"
                response = session.get(url, timeout=TIMEOUT)
                response.raise_for_status()
                passages = parse_open_article_xml(response.content)
                return {"status": "full_text", "provider": "Europe PMC", "url": url,
                        "pmcid": match["pmcid"], "passages": passages,
                        "character_count": sum(len(item["text"]) for item in passages)}
        except (requests.RequestException, ValueError, ElementTree.ParseError, TypeError) as exc:
            xml_error = f"Europe PMC lookup failed: {exc}"
    pdf_url = source.get("oa_pdf_url")
    if pdf_url:
        try:
            content, verified_url = _read_open_pdf(pdf_url, session)
            passages = parse_open_pdf(content, source["title"])
            return {"status": "full_text", "provider": "open-access PDF",
                    "url": verified_url, "license": source.get("oa_license"),
                    "version": source.get("oa_version"), "passages": passages,
                    "character_count": sum(len(item["text"]) for item in passages)}
        except (requests.RequestException, ValueError, OSError) as exc:
            return {"status": "abstract_only", "reason": f"{xml_error} PDF: {exc}"}
    return {"status": "abstract_only", "reason": xml_error}


def enrich_with_open_fulltext(sources, session, limit=22):
    warnings = []
    for source in sources:
        source["fulltext"] = {"status": "abstract_only", "reason": "Lookup not attempted."}
    prioritized = sorted(enumerate(sources), key=lambda pair: (not bool(pair[1].get("oa_pdf_url")), pair[0]))
    for _, source in prioritized[:limit]:
        try:
            source["fulltext"] = fetch_open_fulltext(source, session)
        except (requests.RequestException, ValueError, ElementTree.ParseError, TypeError, OSError) as exc:
            source["fulltext"] = {"status": "abstract_only", "reason": "Open full-text retrieval failed."}
            warnings.append(f"Open full text {source['id']}: {exc}")
    return warnings
