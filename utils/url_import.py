import requests
from bs4 import BeautifulSoup


def fetch_url_text(url: str) -> str:
    """Fetch a URL and return cleaned body text (removes nav/footer/scripts)."""
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    body = soup.find("body")
    text = (body or soup).get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)
