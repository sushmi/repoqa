"""Tree-sitter AST parser registry.

Uses the new tree-sitter >=0.21 Python API where grammar packages expose
a `language()` function returning a `Language` capsule object.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports: only attempt to import grammar packages that are installed.
# If a grammar package is missing, that language falls back to plain-text chunking.

try:
    import tree_sitter_python as _tspy
    _PY_LANGUAGE = _tspy.language()
except ImportError:
    _PY_LANGUAGE = None

try:
    import tree_sitter_javascript as _tsjs
    _JS_LANGUAGE = _tsjs.language()
except ImportError:
    _JS_LANGUAGE = None

try:
    import tree_sitter_typescript as _tsts
    # tree-sitter-typescript exposes two grammars: typescript and tsx
    _TS_LANGUAGE = _tsts.language_typescript()
    _TSX_LANGUAGE = _tsts.language_tsx()
except (ImportError, AttributeError):
    try:
        import tree_sitter_typescript as _tsts  # type: ignore[no-redef]
        _TS_LANGUAGE = _tsts.language()
        _TSX_LANGUAGE = _tsts.language()
    except ImportError:
        _TS_LANGUAGE = None
        _TSX_LANGUAGE = None

try:
    import tree_sitter_java as _tsjava
    _JAVA_LANGUAGE = _tsjava.language()
except ImportError:
    _JAVA_LANGUAGE = None

try:
    import tree_sitter_go as _tsgo
    _GO_LANGUAGE = _tsgo.language()
except ImportError:
    _GO_LANGUAGE = None

try:
    import tree_sitter_rust as _tsrust
    _RUST_LANGUAGE = _tsrust.language()
except ImportError:
    _RUST_LANGUAGE = None

try:
    import tree_sitter_c as _tsc
    _C_LANGUAGE = _tsc.language()
except ImportError:
    _C_LANGUAGE = None

try:
    import tree_sitter_cpp as _tscpp
    _CPP_LANGUAGE = _tscpp.language()
except ImportError:
    _CPP_LANGUAGE = None


# Map from our internal language strings to grammar capsules
_LANGUAGE_CAPSULES: dict[str, object] = {}
for _lang, _capsule in [
    ("python", _PY_LANGUAGE),
    ("javascript", _JS_LANGUAGE),
    ("typescript", _TS_LANGUAGE),
    ("tsx", _TSX_LANGUAGE),
    ("java", _JAVA_LANGUAGE),
    ("go", _GO_LANGUAGE),
    ("rust", _RUST_LANGUAGE),
    ("c", _C_LANGUAGE),
    ("cpp", _CPP_LANGUAGE),
]:
    if _capsule is not None:
        _LANGUAGE_CAPSULES[_lang] = _capsule


class ASTParserRegistry:
    """Manages one tree-sitter Parser per supported language."""

    def __init__(self) -> None:
        try:
            from tree_sitter import Language, Parser
            self._Language = Language
            self._Parser = Parser
            self._available = True
        except ImportError:
            logger.warning("tree-sitter not installed — AST parsing disabled, falling back to text splitting.")
            self._available = False

        self._parsers: dict[str, object] = {}
        if self._available:
            self._init_parsers()

    def _init_parsers(self) -> None:
        for lang, capsule in _LANGUAGE_CAPSULES.items():
            try:
                language_obj = self._Language(capsule)
                parser = self._Parser(language_obj)
                self._parsers[lang] = parser
                logger.debug("Initialized tree-sitter parser for %s", lang)
            except Exception as exc:
                logger.warning("Failed to init tree-sitter parser for %s: %s", lang, exc)

    def supports(self, language: str) -> bool:
        return language in self._parsers

    def parse(self, language: str, source: str):
        """Parse *source* text and return a tree-sitter Tree, or None on failure."""
        if not self._available or language not in self._parsers:
            return None
        parser = self._parsers[language]
        try:
            return parser.parse(source.encode("utf-8"))
        except Exception as exc:
            logger.debug("Parse error for language=%s: %s", language, exc)
            return None

    @staticmethod
    def get_node_text(node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")

    @property
    def supported_languages(self) -> list[str]:
        return list(self._parsers.keys())
