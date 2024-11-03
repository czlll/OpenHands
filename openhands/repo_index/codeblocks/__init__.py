from typing import Optional

from openhands.repo_index.codeblocks.parser.java import JavaParser
from openhands.repo_index.codeblocks.parser.parser import CodeParser
from openhands.repo_index.codeblocks.parser.python import PythonParser


def supports_codeblocks(path: str):
    return path.endswith('.py')


def get_parser_by_path(file_path: str) -> Optional[CodeParser]:
    if file_path.endswith('.py'):
        return PythonParser()
    elif file_path.endswith('.java'):
        return JavaParser()
    else:
        return None
