"""Dependency graph builder: extracts import relationships from AST.

Constructs a directed graph G = (V, E) where:
  - V = source files (repo_path)
  - E = import relationships (file A imports file B)

Uses tree-sitter to find import nodes, then resolves module paths
to actual files in the repository.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path, PurePosixPath

from repoqa.models import FileRecord
from repoqa.ingestion.ast_parser import ASTParserRegistry

logger = logging.getLogger(__name__)

# Import node types per language
_IMPORT_NODE_TYPES = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "java": {"import_declaration"},
    "go": {"import_declaration", "import_spec"},
    "rust": {"use_declaration"},
    "c": {"preproc_include"},
    "cpp": {"preproc_include"},
}


@dataclass
class DepGraph:
    """Directed import dependency graph."""

    # file_path → list of imported file paths
    edges: dict[str, list[str]] = field(default_factory=dict)
    # file_path → list of raw import strings (unresolved)
    raw_imports: dict[str, list[str]] = field(default_factory=dict)

    @property
    def nodes(self) -> set[str]:
        all_nodes = set(self.edges.keys())
        for targets in self.edges.values():
            all_nodes.update(targets)
        return all_nodes

    def get_imports(self, file_path: str) -> list[str]:
        """Files that file_path imports (outgoing edges)."""
        return self.edges.get(file_path, [])

    def get_importers(self, file_path: str) -> list[str]:
        """Files that import file_path (incoming edges)."""
        return [src for src, targets in self.edges.items() if file_path in targets]

    def get_neighbors(self, file_path: str) -> list[str]:
        """All files connected to file_path (imports + importers)."""
        neighbors = set(self.get_imports(file_path))
        neighbors.update(self.get_importers(file_path))
        neighbors.discard(file_path)
        return list(neighbors)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"edges": self.edges, "raw_imports": self.raw_imports}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> DepGraph:
        with open(path) as f:
            data = json.load(f)
        return cls(edges=data["edges"], raw_imports=data["raw_imports"])

    def summary(self) -> dict:
        total_edges = sum(len(v) for v in self.edges.values())
        return {
            "files": len(self.edges),
            "total_edges": total_edges,
            "avg_imports_per_file": round(total_edges / max(len(self.edges), 1), 1),
        }


class DepGraphBuilder:
    """Extracts import relationships from parsed ASTs and resolves to repo paths."""

    def __init__(self) -> None:
        self._parser = ASTParserRegistry()

    def build(self, file_records: list[FileRecord]) -> DepGraph:
        """Build a dependency graph from a list of file records."""
        # Index of all known file paths for resolution
        known_paths = {r.repo_path for r in file_records}

        graph = DepGraph()

        for record in file_records:
            if record.language not in _IMPORT_NODE_TYPES:
                continue

            tree = self._parser.parse(record.language, record.raw_content)
            if tree is None:
                continue

            raw_imports = self._extract_imports(
                tree, record.raw_content.encode("utf-8"), record.language
            )
            graph.raw_imports[record.repo_path] = raw_imports

            resolved = []
            for imp in raw_imports:
                target = self._resolve_import(
                    imp, record.repo_path, record.language, known_paths
                )
                if target:
                    resolved.append(target)

            if resolved:
                graph.edges[record.repo_path] = resolved

        total_edges = sum(len(v) for v in graph.edges.values())
        logger.info(
            "Dependency graph: %d files, %d edges", len(graph.edges), total_edges
        )
        return graph

    def _extract_imports(
        self, tree, source_bytes: bytes, language: str
    ) -> list[str]:
        """Walk the AST and extract raw import module strings."""
        target_types = _IMPORT_NODE_TYPES.get(language, set())
        imports: list[str] = []

        for node in tree.root_node.children:
            if node.type not in target_types:
                continue

            if language == "python":
                imp = self._extract_python_import(node, source_bytes)
            elif language in ("javascript", "typescript"):
                imp = self._extract_js_import(node, source_bytes)
            elif language == "java":
                imp = self._extract_java_import(node, source_bytes)
            elif language == "go":
                imp = self._extract_go_import(node, source_bytes)
            elif language in ("c", "cpp"):
                imp = self._extract_c_include(node, source_bytes)
            elif language == "rust":
                imp = self._extract_rust_use(node, source_bytes)
            else:
                imp = None

            if imp:
                imports.append(imp)

        return imports

    # ---- Language-specific import extractors ----

    def _extract_python_import(self, node, source_bytes: bytes) -> str | None:
        """Extract module path from import_statement or import_from_statement."""
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    return self._node_text(child, source_bytes)
        elif node.type == "import_from_statement":
            # from X import Y → we want X
            for child in node.children:
                if child.type == "dotted_name":
                    return self._node_text(child, source_bytes)
                if child.type == "relative_import":
                    return self._node_text(child, source_bytes)
        return None

    def _extract_js_import(self, node, source_bytes: bytes) -> str | None:
        """Extract path from: import X from 'path' or import 'path'."""
        for child in node.children:
            if child.type == "string":
                raw = self._node_text(child, source_bytes)
                return raw.strip("'\"")
        return None

    def _extract_java_import(self, node, source_bytes: bytes) -> str | None:
        """Extract package path from: import com.foo.Bar;"""
        for child in node.children:
            if child.type == "scoped_identifier":
                return self._node_text(child, source_bytes)
        return None

    def _extract_go_import(self, node, source_bytes: bytes) -> str | None:
        """Extract path from: import "fmt" or import "github.com/foo/bar"."""
        for child in node.children:
            if child.type == "interpreted_string_literal":
                raw = self._node_text(child, source_bytes)
                return raw.strip('"')
        return None

    def _extract_c_include(self, node, source_bytes: bytes) -> str | None:
        """Extract path from: #include "file.h" or #include <stdio.h>."""
        for child in node.children:
            if child.type in ("string_literal", "system_lib_string"):
                raw = self._node_text(child, source_bytes)
                return raw.strip('"<>')
        return None

    def _extract_rust_use(self, node, source_bytes: bytes) -> str | None:
        """Extract crate path from: use std::collections::HashMap;"""
        for child in node.children:
            if child.type in ("scoped_identifier", "use_wildcard", "scoped_use_list"):
                return self._node_text(child, source_bytes)
        return None

    # ---- Import resolution ----

    def _resolve_import(
        self,
        raw_import: str,
        source_path: str,
        language: str,
        known_paths: set[str],
    ) -> str | None:
        """Try to resolve a raw import string to an actual file in the repo."""
        candidates = self._generate_candidates(raw_import, source_path, language)
        for candidate in candidates:
            # Normalize to forward slashes
            normalized = str(PurePosixPath(candidate))
            if normalized in known_paths:
                return normalized
        return None

    def _generate_candidates(
        self, raw_import: str, source_path: str, language: str
    ) -> list[str]:
        """Generate possible file paths from a raw import string."""
        source_dir = str(PurePosixPath(source_path).parent)

        if language == "python":
            return self._python_candidates(raw_import, source_dir)
        elif language in ("javascript", "typescript"):
            return self._js_candidates(raw_import, source_dir)
        elif language == "java":
            return self._java_candidates(raw_import)
        elif language in ("c", "cpp"):
            return self._c_candidates(raw_import, source_dir)
        elif language == "go":
            return self._go_candidates(raw_import, source_dir)
        return []

    def _python_candidates(self, raw: str, source_dir: str) -> list[str]:
        """from flask.views → flask/views.py, flask/views/__init__.py"""
        # Handle relative imports (starts with .)
        if raw.startswith("."):
            dots = len(raw) - len(raw.lstrip("."))
            module = raw.lstrip(".")
            base = source_dir
            for _ in range(dots - 1):
                base = str(PurePosixPath(base).parent)
            path_part = module.replace(".", "/")
            if path_part:
                return [
                    f"{base}/{path_part}.py",
                    f"{base}/{path_part}/__init__.py",
                ]
            return [f"{base}/__init__.py"]

        # Absolute import
        path_part = raw.replace(".", "/")
        return [
            f"{path_part}.py",
            f"{path_part}/__init__.py",
            f"src/{path_part}.py",
            f"src/{path_part}/__init__.py",
        ]

    def _js_candidates(self, raw: str, source_dir: str) -> list[str]:
        """import X from './App' → src/App.js, src/App.ts, src/App/index.js"""
        if not raw.startswith("."):
            return []  # skip node_modules (external packages)

        # Resolve .. and . in the path
        base = str(PurePosixPath(source_dir, raw))
        # Normalize to remove ../
        parts = []
        for part in base.split("/"):
            if part == "..":
                if parts:
                    parts.pop()
            elif part != ".":
                parts.append(part)
        base = "/".join(parts)

        return [
            f"{base}.js",
            f"{base}.ts",
            f"{base}.tsx",
            f"{base}.jsx",
            f"{base}/index.js",
            f"{base}/index.ts",
            base,
        ]

    def _java_candidates(self, raw: str) -> list[str]:
        """import com.foo.Bar → com/foo/Bar.java, src/main/java/com/foo/Bar.java"""
        path_part = raw.replace(".", "/")
        return [
            f"{path_part}.java",
            f"src/main/java/{path_part}.java",
        ]

    def _c_candidates(self, raw: str, source_dir: str) -> list[str]:
        """#include "utils.h" → src/utils.h, utils.h"""
        return [
            f"{source_dir}/{raw}",
            raw,
            f"src/{raw}",
            f"include/{raw}",
        ]

    def _go_candidates(self, raw: str, source_dir: str) -> list[str]:
        """Go imports are package paths — hard to resolve without go.mod."""
        # Only resolve relative-looking paths
        parts = raw.split("/")
        if len(parts) >= 2:
            return [f"{'/'.join(parts)}.go"]
        return []

    @staticmethod
    def _node_text(node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
