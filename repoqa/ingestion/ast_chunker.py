"""AST-aware chunker: splits code files into semantically meaningful Chunk objects.

Each chunk corresponds to a complete syntactic unit (function, class, module-level
code) rather than an arbitrary character window.

Tree-sitter online documenation - https://tree-sitter.github.io/
"""

from __future__ import annotations

import logging

from repoqa.models import Chunk, FileRecord
from repoqa.ingestion.ast_parser import ASTParserRegistry
from repoqa.tokenizer import count_tokens as _count_tokens

logger = logging.getLogger(__name__)


# Node types that represent top-level structural units per language.
# We collect these as the primary chunk boundaries.
_TOP_LEVEL_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
        "async_function_definition",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "arrow_function",
        "export_statement",
        "method_definition",
        "function_expression",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "export_statement",
        "method_definition",
        "arrow_function",
    },
    "java": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
        "enum_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "var_declaration",
        "const_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "trait_item",
        "enum_item",
        "mod_item",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
        "template_declaration",
    },
}

# Node type → human-readable chunk_type string
_TYPE_TO_CHUNK_TYPE: dict[str, str] = {
    "function_definition": "function",
    "async_function_definition": "function",
    "function_declaration": "function",
    "arrow_function": "function",
    "function_expression": "function",
    "function_item": "function",
    "method_declaration": "function",
    "method_definition": "function",
    "constructor_declaration": "function",
    "class_definition": "class",
    "class_declaration": "class",
    "class_specifier": "class",
    "interface_declaration": "class",
    "type_alias_declaration": "class",
    "impl_item": "class",
    "struct_item": "class",
    "struct_specifier": "class",
    "trait_item": "class",
    "enum_item": "class",
    "enum_declaration": "class",
    "enum_specifier": "class",
    "mod_item": "module",
    "namespace_definition": "module",
}


def _extract_symbol_name(node, source_bytes: bytes, language: str) -> str | None:
    """Try to extract the name identifier from a structural node."""
    # Walk direct children looking for an "identifier" or "name" node
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier"):
            return source_bytes[child.start_byte: child.end_byte].decode("utf-8", errors="replace")
    return None


class ASTChunker:
    """Converts a source file into a list of Chunk objects via AST parsing.

    Falls back to a simple line-based splitter for files where the parser
    is unavailable or fails.
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._registry = ASTParserRegistry()

    def chunk_file(self, file_record: FileRecord) -> list[Chunk]:
        """Main entry point: parse and chunk a single file."""
        language = file_record.language

        # If we have a parser, use AST-aware chunking
        if self._registry.supports(language):
            tree = self._registry.parse(language, file_record.raw_content)
            if tree is not None:
                return self._ast_chunk(file_record, tree)

        # Fallback: line-based splitting
        logger.debug("Falling back to line-based chunking for %s", file_record.repo_path)
        return self._line_chunk(file_record)

    # ------------------------------------------------------------------
    # AST-aware chunking
    # ------------------------------------------------------------------

    def _ast_chunk(self, file_record: FileRecord, tree) -> list[Chunk]:
        source_bytes = file_record.raw_content.encode("utf-8")
        language = file_record.language
        target_types = _TOP_LEVEL_TYPES.get(language, set())

        chunks: list[Chunk] = []
        covered_lines: set[int] = set()

        root = tree.root_node
        for node in root.children:
            # Unwrap export_statement to get the actual declaration
            inner = node
            if node.type == "export_statement":
                for child in node.children:
                    if child.type in target_types:
                        inner = child
                        break

            if inner.type not in target_types:
                continue

            node_chunks = self._node_to_chunks(inner, source_bytes, file_record, covered_lines)
            chunks.extend(node_chunks)

        # Collect uncovered lines as "module" chunks (imports, globals, etc.)
        module_chunks = self._collect_module_chunks(file_record, covered_lines)
        chunks.extend(module_chunks)

        return chunks if chunks else self._line_chunk(file_record)

    def _node_to_chunks(
        self,
        node,
        source_bytes: bytes,
        file_record: FileRecord,
        covered_lines: set[int],
    ) -> list[Chunk]:
        text = ASTParserRegistry.get_node_text(node, source_bytes)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        token_count = _count_tokens(text)

        # Mark these lines as covered
        covered_lines.update(range(start_line, end_line + 1))

        if token_count <= self.max_tokens:
            chunk_type = _TYPE_TO_CHUNK_TYPE.get(node.type, "module")
            symbol_name = _extract_symbol_name(node, source_bytes, file_record.language)
            return [
                Chunk(
                    id=Chunk.make_id(file_record.repo_path, start_line),
                    file_record=file_record,
                    content=text,
                    chunk_type=chunk_type,
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=symbol_name,
                    token_count=token_count,
                )
            ]
        else:
            # Node is too large: split by children
            return self._split_oversized_node(node, source_bytes, file_record)

    def _split_oversized_node(self, node, source_bytes: bytes, file_record: FileRecord) -> list[Chunk]:
        """Recursively split a node that exceeds max_tokens."""
        chunks: list[Chunk] = []
        buffer_lines: list[str] = []
        buffer_start = node.start_point[0] + 1
        prev_tail: str = ""

        for child in node.children:
            child_text = ASTParserRegistry.get_node_text(child, source_bytes)
            child_lines = child_text.splitlines()
            prospective = (prev_tail + "\n" if prev_tail else "") + child_text
            if _count_tokens(prospective) > self.max_tokens and buffer_lines:
                # Flush buffer as a chunk
                content = "\n".join(buffer_lines)
                start_line = buffer_start
                end_line = buffer_start + len(buffer_lines) - 1
                chunks.append(
                    Chunk(
                        id=Chunk.make_id(file_record.repo_path, start_line),
                        file_record=file_record,
                        content=content,
                        chunk_type="module",
                        start_line=start_line,
                        end_line=end_line,
                        symbol_name=None,
                        token_count=_count_tokens(content),
                    )
                )
                # Start new buffer with overlap from previous tail
                overlap_lines = prev_tail.splitlines()[-self.overlap_tokens // 10:] if prev_tail else []
                buffer_lines = overlap_lines + child_lines
                buffer_start = child.start_point[0] + 1
            else:
                buffer_lines.extend(child_lines)

            # Keep the last ~overlap_tokens worth of text as context for the next chunk
            tail_text = "\n".join(buffer_lines)
            tail_tokens = _count_tokens(tail_text)
            if tail_tokens > self.overlap_tokens:
                # Trim to last overlap_tokens tokens worth of lines
                tail_lines = buffer_lines
                while _count_tokens("\n".join(tail_lines)) > self.overlap_tokens and len(tail_lines) > 1:
                    tail_lines = tail_lines[1:]
                prev_tail = "\n".join(tail_lines)
            else:
                prev_tail = tail_text

        # Flush any remaining buffer
        if buffer_lines:
            content = "\n".join(buffer_lines)
            start_line = buffer_start
            end_line = buffer_start + len(buffer_lines) - 1
            chunks.append(
                Chunk(
                    id=Chunk.make_id(file_record.repo_path, start_line),
                    file_record=file_record,
                    content=content,
                    chunk_type="module",
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=None,
                    token_count=_count_tokens(content),
                )
            )

        return chunks

    def _collect_module_chunks(self, file_record: FileRecord, covered_lines: set[int]) -> list[Chunk]:
        """Collect source lines not covered by any structural node into module chunks."""
        lines = file_record.raw_content.splitlines()
        uncovered: list[tuple[int, str]] = [
            (i + 1, line)
            for i, line in enumerate(lines)
            if (i + 1) not in covered_lines
        ]
        if not uncovered:
            return []

        # Group contiguous uncovered lines
        groups: list[list[tuple[int, str]]] = []
        current: list[tuple[int, str]] = [uncovered[0]]
        for lineno, text in uncovered[1:]:
            if lineno == current[-1][0] + 1:
                current.append((lineno, text))
            else:
                groups.append(current)
                current = [(lineno, text)]
        groups.append(current)

        chunks: list[Chunk] = []
        for group in groups:
            content = "\n".join(t for _, t in group)
            content = content.strip()
            if not content:
                continue
            start_line = group[0][0]
            end_line = group[-1][0]
            # Further split if group exceeds max_tokens
            for sub_chunk in self._sliding_window(content, start_line, file_record, "module"):
                chunks.append(sub_chunk)

        return chunks

    # ------------------------------------------------------------------
    # Fallback: line-based sliding window
    # ------------------------------------------------------------------

    def _line_chunk(self, file_record: FileRecord) -> list[Chunk]:
        content = file_record.raw_content
        return self._sliding_window(content, 1, file_record, "prose")

    def _sliding_window(
        self,
        content: str,
        start_line_offset: int,
        file_record: FileRecord,
        chunk_type: str,
    ) -> list[Chunk]:
        """Split *content* into overlapping token-budget-respecting chunks."""
        lines = content.splitlines()
        chunks: list[Chunk] = []
        i = 0
        while i < len(lines):
            batch: list[str] = []
            token_count = 0
            j = i
            while j < len(lines) and token_count + _count_tokens(lines[j]) <= self.max_tokens:
                batch.append(lines[j])
                token_count += _count_tokens(lines[j])
                j += 1
            if not batch:
                # Single line exceeds max_tokens — include it anyway
                batch = [lines[i]]
                j = i + 1

            chunk_content = "\n".join(batch)
            abs_start = start_line_offset + i
            abs_end = start_line_offset + j - 1
            chunks.append(
                Chunk(
                    id=Chunk.make_id(file_record.repo_path, abs_start),
                    file_record=file_record,
                    content=chunk_content,
                    chunk_type=chunk_type,
                    start_line=abs_start,
                    end_line=abs_end,
                    symbol_name=None,
                    token_count=_count_tokens(chunk_content),
                )
            )
            # Advance with overlap
            overlap_lines = max(1, self.overlap_tokens // 10)
            i = j - overlap_lines if j - overlap_lines > i else j

        return chunks
