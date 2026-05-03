from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field

from repo_agent.tools.base import BaseTool, ToolResult


class TraceSymbolArgs(BaseModel):
    symbol_name: str
    path: str = "."
    max_results: int = Field(default=20, ge=1)


class _SymbolOccurrence(BaseModel):
    path: str
    line: int
    column: int
    occurrence_type: str
    symbol_kind: str
    code_line: str


class _SymbolTraceVisitor(ast.NodeVisitor):
    def __init__(self, symbol_name: str, source_lines: list[str]) -> None:
        self.symbol_name = symbol_name
        self.source_lines = source_lines
        self.occurrences: list[_SymbolOccurrence] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.symbol_name:
            self._add(node, "definition", "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == self.symbol_name:
            self._add(node, "definition", "async_function")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == self.symbol_name:
            self._add(node, "definition", "class")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == self.symbol_name or alias.asname == self.symbol_name:
                self._add(node, "usage", "import")
                break
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == self.symbol_name or alias.asname == self.symbol_name:
                self._add(node, "usage", "import_from")
                break
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._visit_assignment_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_assignment_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._visit_assignment_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_assignment_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_assignment_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._visit_assignment_target(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._visit_assignment_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name == self.symbol_name:
            self._add(node, "definition", "exception_variable")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.symbol_name and isinstance(node.ctx, ast.Load):
            self._add(node, "usage", "name")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == self.symbol_name:
            self._add(node, "usage", "attribute")
        self.generic_visit(node)

    def _visit_assignment_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name) and target.id == self.symbol_name:
            self._add(target, "definition", "variable")
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._visit_assignment_target(element)

    def _add(self, node: ast.AST, occurrence_type: str, symbol_kind: str) -> None:
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", 0)
        if lineno is None or lineno <= 0 or lineno > len(self.source_lines):
            return
        self.occurrences.append(
            _SymbolOccurrence(
                path="",
                line=lineno,
                column=col_offset + 1,
                occurrence_type=occurrence_type,
                symbol_kind=symbol_kind,
                code_line=self.source_lines[lineno - 1].rstrip(),
            )
        )


class TraceSymbolTool(BaseTool):
    name = "trace_symbol"
    description = "Trace symbol definitions and usage locations in Python files via AST."
    args_model = TraceSymbolArgs

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        args = TraceSymbolArgs.model_validate(arguments)
        if not args.symbol_name.strip():
            return ToolResult(success=False, content="symbol_name must not be empty")

        try:
            target = self._resolve_repo_path(args.path)
        except ValueError as exc:
            return ToolResult(success=False, content=str(exc))

        if not target.exists():
            return ToolResult(success=False, content=f"path does not exist: {args.path}")

        occurrences: list[_SymbolOccurrence] = []
        for file_path in self._iter_python_files(target):
            occurrences.extend(self._trace_file(file_path, args.symbol_name))

        occurrences.sort(key=lambda item: (item.path, item.line, item.column))
        truncated = len(occurrences) > args.max_results
        visible = occurrences[: args.max_results]

        if not visible:
            return ToolResult(
                success=True,
                content=f"No occurrences found for symbol `{args.symbol_name}`.",
                metadata={
                    "symbol_name": args.symbol_name,
                    "match_count": 0,
                    "truncated": False,
                    "occurrences": [],
                },
            )

        lines = [
            f"{item.path}:{item.line}:{item.column}: {item.occurrence_type} ({item.symbol_kind}): {item.code_line}"
            for item in visible
        ]
        return ToolResult(
            success=True,
            content="\n".join(lines),
            metadata={
                "symbol_name": args.symbol_name,
                "match_count": len(occurrences),
                "truncated": truncated,
                "occurrences": [item.model_dump() for item in visible],
            },
        )

    def _trace_file(self, file_path: Path, symbol_name: str) -> list[_SymbolOccurrence]:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return []

        source_lines = source.splitlines()
        visitor = _SymbolTraceVisitor(symbol_name, source_lines)
        visitor.visit(tree)

        rel_path = file_path.relative_to(self.repo_root).as_posix()
        for occurrence in visitor.occurrences:
            occurrence.path = rel_path
        return visitor.occurrences

    def _iter_python_files(self, target: Path) -> list[Path]:
        if target.is_file():
            return [target] if target.suffix == ".py" else []
        return sorted(path for path in target.rglob("*.py") if path.is_file())

    def _resolve_repo_path(self, raw_path: str) -> Path:
        candidate = (self.repo_root / raw_path).resolve()
        if self.repo_root != candidate and self.repo_root not in candidate.parents:
            raise ValueError(f"path escapes repository root: {raw_path}")
        return candidate
