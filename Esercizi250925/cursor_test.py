import pytest
from app_functions import slugify, internet_research


def test_slugify_basic():
    # basic conversion
    assert slugify("Hello World") == "hello-world"

    # multiple spaces are converted to multiple hyphens (by current implementation)
    assert slugify("Multiple   Spaces") == "multiple---spaces"

    # existing hyphens are preserved
    assert slugify("Already-slugified") == "already-slugified"

    # leading/trailing spaces become hyphens
    assert slugify(" Trailing space ") == "-trailing-space-"

    # empty string remains empty
    assert slugify("") == ""

    # non-ascii characters are lowercased but kept as-is
    assert slugify("Caffè Latte") == "caffè-latte"


def test_slugify_edge_cases():
    # only spaces -> only hyphens
    assert slugify("   ") == "---"

    # tabs/newlines are preserved; only spaces become hyphens
    assert slugify("\tTabbed Text\t") == "\ttabbed-text\t"
    assert slugify("Line\nBreak") == "line\nbreak"

    # multiple consecutive spaces -> multiple hyphens
    assert slugify("Multiple    spaces") == "multiple----spaces"

    # numbers and punctuation are preserved (only spaces replaced)
    assert slugify("123 ABC!@#") == "123-abc!@#"

    # very long strings maintain length structure; single space becomes single hyphen
    long_input = ("A" * 1000) + " " + ("B" * 1000)
    long_output = slugify(long_input)
    assert long_output.startswith("a" * 1000)
    assert long_output.endswith("b" * 1000)
    assert long_output.count("-") == 1

    # non-string input raises (current implementation calls .lower on input)
    with pytest.raises(AttributeError):
        slugify(None)
    
def test_internet_research_returns_list_and_prints_count(monkeypatch, capsys):
    # Arrange: inject a fake 'ddgs' module
    import sys
    from types import ModuleType

    class FakeDDGS:
        def __init__(self, verify=False):
            self.verify = verify

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=10):
            return iter([
                {"title": "Result 1", "url": "http://example.com/1"},
                {"title": "Result 2", "url": "http://example.com/2"},
            ])

    fake_module = ModuleType("ddgs")
    fake_module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)

    # Act
    results = internet_research("pytest")

    # Assert
    assert isinstance(results, list)
    assert len(results) == 2
    captured = capsys.readouterr()
    assert "Found 2 results for query: pytest" in captured.out


def test_internet_research_empty_results(monkeypatch, capsys):
    import sys
    from types import ModuleType

    class FakeDDGS:
        def __init__(self, verify=False):
            self.verify = verify

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=10):
            return iter([])

    fake_module = ModuleType("ddgs")
    fake_module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)

    results = internet_research("")
    assert results == []
    captured = capsys.readouterr()
    assert "Found 0 results for query: " in captured.out


def test_internet_research_raises_on_ddgs_error(monkeypatch):
    import sys
    from types import ModuleType

    class FakeDDGS:
        def __init__(self, verify=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=10):
            raise RuntimeError("ddgs failure")

    fake_module = ModuleType("ddgs")
    fake_module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)

    with pytest.raises(RuntimeError, match="ddgs failure"):
        internet_research("anything")