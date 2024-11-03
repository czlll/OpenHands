import difflib
import glob
import logging
import os
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from openhands.repo_index.codeblocks import (
    get_parser_by_path,
)
from openhands.repo_index.codeblocks.module import (
    Module,
)
from openhands.repo_index.codeblocks.parser.python import (
    PythonParser,
)

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    file_path: str
    updated: bool
    diff: Optional[str] = None
    error: Optional[str] = None
    new_span_ids: Optional[set[str]] = None


class CodeFile(BaseModel):
    file_path: str
    content: str
    module: Optional[Module] = None

    dirty: bool = False

    @classmethod
    def from_file(cls, repo_path: str, file_path: str):
        with open(os.path.join(repo_path, file_path), 'r') as f:
            parser = get_parser_by_path(file_path)
            if parser:
                content = f.read()
                module = parser.parse(content)
            else:
                module = None
            return cls(file_path=file_path, content=content, module=module)

    @classmethod
    def from_content(cls, file_path: str, content: str):
        parser = PythonParser()
        module = parser.parse(content)
        return cls(file_path=file_path, content=content, module=module)

    @property
    def supports_codeblocks(self):
        return self.module is not None


class FileRepository:
    def __init__(self, repo_path: str):
        self._repo_path = repo_path
        self._files: dict[str, CodeFile] = {}

    @property
    def path(self):
        return self._repo_path

    def get_file(
        self, file_path: str, refresh: bool = False, from_origin: bool = False
    ):
        """
        Get a file from the repository.

        Args:

        """
        file = self._files.get(file_path)
        if not file or refresh or from_origin:
            full_file_path = os.path.join(self._repo_path, file_path)
            if not os.path.exists(full_file_path):
                logger.warning(f'File not found: {full_file_path}')
                return None
            if not os.path.isfile(full_file_path):
                logger.warning(f'{full_file_path} is not a file')
                return None

            with open(full_file_path, 'r') as f:
                parser = get_parser_by_path(file_path)
                if parser:
                    content = f.read()
                    module = parser.parse(content)
                    file = CodeFile(file_path=file_path, content=content, module=module)
                else:
                    file = CodeFile(file_path=file_path, content=f.read())

            if refresh or not from_origin:
                self._files[file_path] = file
        return file

    def matching_files(self, file_pattern: str):
        matched_files = []
        for matched_file in glob.iglob(
            file_pattern, root_dir=self._repo_path, recursive=True
        ):
            matched_files.append(matched_file)

        if not matched_files and not file_pattern.startswith('*'):
            return self.matching_files(f'**/{file_pattern}')

        return matched_files

    def find_files(self, file_patterns: list[str]) -> set[str]:
        found_files = set()
        for file_pattern in file_patterns:
            matched_files = self.matching_files(file_pattern)
            found_files.update(matched_files)

        return found_files

    def has_matching_files(self, file_pattern: str):
        for matched_file in glob.iglob(
            file_pattern, root_dir=self._repo_path, recursive=True
        ):
            return True
        return False

    def file_match(self, file_pattern: str, file_path: str):
        match = False
        for matched_file in glob.iglob(
            file_pattern, root_dir=self._repo_path, recursive=True
        ):
            if matched_file == file_path:
                match = True
                break
        return match


def remove_duplicate_lines(replacement_lines, original_lines):
    """
    Removes overlapping lines at the end of replacement_lines that match the beginning of original_lines.
    """
    if not replacement_lines or not original_lines:
        return replacement_lines

    max_overlap = min(len(replacement_lines), len(original_lines))

    for overlap in range(max_overlap, 0, -1):
        if replacement_lines[-overlap:] == original_lines[:overlap]:
            return replacement_lines[:-overlap]

    return replacement_lines


def do_diff(
    file_path: str, original_content: str, updated_content: str
) -> Optional[str]:
    return ''.join(
        difflib.unified_diff(
            original_content.strip().splitlines(True),
            updated_content.strip().splitlines(True),
            fromfile=file_path,
            tofile=file_path,
            lineterm='\n',
        )
    )
