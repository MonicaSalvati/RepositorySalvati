import re
import unicodedata
from typing import Optional

def slugify(text):
    """
    Convert text to a slug by lowercasing and replacing spaces with hyphens.

    Parameters
    ----------
    text : str
        Input string to slugify.

    Returns
    -------
    str
        Slugified string where spaces are replaced by '-' and letters are lowercased.

    Examples
    --------
    >>> slugify("Hello World")
    'hello-world'
    """
    return text.lower().replace(" ", "-")


_slug_re = re.compile(r"[^a-zA-Z0-9]+", re.UNICODE)
_slug_re_unicode = re.compile(r"\W+", re.UNICODE)


def slugify_advanced(
    text: str,
    *,
    separator: str = "-",
    max_length: Optional[int] = None,
    preserve_unicode: bool = False,
    lowercase: bool = True,
) -> str:
    """
    Convert text to a URL-safe slug with normalization and options.

    Parameters
    ----------
    text : str
        Input string to slugify.
    separator : str, optional
        String used to join tokens. Default is "-".
    max_length : int or None, optional
        If provided, truncate the slug to this length (without trailing separator).
    preserve_unicode : bool, optional
        If True, keep non-ASCII letters; otherwise, strip accents to ASCII. Default False.
    lowercase : bool, optional
        If True, lowercase the output. Default True.

    Returns
    -------
    str
        URL-safe slug.
    """
    if not isinstance(text, str):
        text = str(text)

    value = text.strip()
    if lowercase:
        value = value.lower()

    if preserve_unicode:
        value = _slug_re_unicode.sub(separator, value)
    else:
        value = unicodedata.normalize("NFKD", value)
        value = value.encode("ascii", "ignore").decode("ascii")
        value = _slug_re.sub(separator, value)

    if len(separator) == 0:
        value = re.sub(r"[^a-zA-Z0-9]+", "", value)
    else:
        sep_escaped = re.escape(separator)
        value = re.sub(fr"{sep_escaped}+", separator, value)
        value = value.strip(separator)

    if max_length is not None and max_length > 0:
        value = value[:max_length].rstrip(separator)

    return value


def internet_research(query):
    """
    Perform an internet search using DuckDuckGo (DDGS).

    Parameters
    ----------
    query : str
        The search query to look up.

    Returns
    -------
    list of dict
        A list of result dictionaries returned by DDGS. Each item may contain
        keys like 'title', 'href'/'url', and 'body' depending on DDGS response.

    Notes
    -----
    - SSL verification is disabled (verify=False) to avoid SSL issues in some environments.
    - This function prints the total number of results found for quick inspection.

    Examples
    --------
    >>> results = internet_research("python slugify")
    >>> isinstance(results, list)
    True
    """
    from ddgs import DDGS
    with DDGS(verify=False) as ddgs:
        results = list(ddgs.text(query, max_results=10))
        print(f"Found {len(results)} results for query: {query}")
        return results


