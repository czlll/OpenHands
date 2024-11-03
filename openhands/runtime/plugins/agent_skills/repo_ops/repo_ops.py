"""repo_ops.py

This module provides various file manipulation skills for the agent.

Functions:
- get_repo_structure(): Get the structure of the repository where the issue resides.
- get_directory_structure(dir_path: str):Search and display the directory structure of a given repository, starting from a specified directory path.
- get_file_content(file_path: str): Get the content of the given file.
- get_file_structures(file_list: list[str]): Get the skeleton/structure of the given files, which you can get all the classes and functions in these files.
- search_class(class_name: str, file_pattern: Optional[str] = None): Searching for the specified class name within the codebase and retrieves its implementation.
- search_class_structures(class_names: list[str], file_pattern: Optional[str] = None): Retrieves class definitions (skeletons) from the codebase based on the provided class names, filtered by an optional file pattern.
- search_method(method_name: str, class_name: Optional[str] = None, file_pattern: Optional[str] = None): Searches for specified method name within the codebase and retrieves its definitions along with relevant context.
- search_invoke_and_reference(module_name: str, file_path: str, ntype: str): Analyzes the dependencies of a specific class, or function, identifying how they interact with other parts of the codebase. This is useful for understanding the broader impact of any changes or issues and ensuring that all related components are accounted for.
- search_interactions_among_modules(module_ids: list[str]): Analyze the interactions between specified modules by examining a pre-built dependency graph.
- search_call_stacks(main_entry: str):
- search_in_repo(search_terms: list[str], file_pattern: Optional[str] = "**/*.py"): Performs a combined search using both the BM25 algorithm and semantic search on the codebase.
"""

import logging
import os
import pickle
import re
import subprocess
import uuid
from copy import deepcopy
from typing import Dict, Optional

# from xmlrpc.client import boolean
import networkx as nx
import Stemmer
from openhands.runtime.plugins.agent_skills.repo_ops.utils.compress_file import get_skeleton
from openhands.runtime.plugins.agent_skills.repo_ops.utils.graph_encoder import RepoSearcher
from openhands.repo_index import Workspace
from openhands.runtime.plugins.agent_skills.repo_ops.utils.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    line_wrap_content,
    show_project_structure,
)
from llama_index.retrievers.bm25 import BM25Retriever
from openhands.runtime.plugins.agent_skills.repo_ops.utils.util import (
    DEPENDENCY_GRAPH_LOC,
    INDEX_STORE_LOC,
    find_matching_files_from_list,
    get_meta_data,
    get_repo_dir_name,
    get_repo_structures,
    setup_full_swebench_repo,
)

logger = logging.getLogger(__name__)

CURRENT_ISSUE_ID: str | None = None
CURRENT_INSTANCE: dict | None = None
CURRENT_STRUCTURE: dict | None = None
ALL_FILE: list | None = None
ALL_CLASS: list | None = None
ALL_FUNC: list | None = None
REPO_SAVE_DIR: str | None = None

FOUND_MODULES: list[str] = []

def add_found_modules(file_path: str, module_name: str, ntype: str = 'file'):
    global FOUND_MODULES
    if ntype == 'file':
        module_id = f'{file_path}'
    else:
        module_id = f'{file_path}:{module_name}'
    if module_id not in FOUND_MODULES:
        FOUND_MODULES.append(module_id)


def get_found_modules():
    global FOUND_MODULES
    return FOUND_MODULES


def set_current_issue(
    instance_id: Optional[str] = None,
    instance_data: Optional[dict] = None,
    dataset: str = 'princeton-nlp/SWE-bench_Lite',
    split: str = 'test',
    rank=0,
):
    global CURRENT_ISSUE_ID, CURRENT_INSTANCE, CURRENT_STRUCTURE
    global ALL_FILE, ALL_CLASS, ALL_FUNC
    assert instance_id or instance_data

    if instance_id:
        CURRENT_ISSUE_ID = instance_id
        CURRENT_INSTANCE = get_meta_data(CURRENT_ISSUE_ID, dataset, split)
    elif instance_data:
        CURRENT_ISSUE_ID = instance_data['instance_id']
        CURRENT_INSTANCE = instance_data

    CURRENT_STRUCTURE = get_repo_structures(CURRENT_INSTANCE)
    ALL_FILE, ALL_CLASS, ALL_FUNC = get_full_file_paths_and_classes_and_functions(
        CURRENT_STRUCTURE
    )

    global REPO_SAVE_DIR
    # Generate a temperary folder and add uuid to avoid collision
    REPO_SAVE_DIR = os.path.join('playground', str(uuid.uuid4()))
    # assert playground doesn't exist
    assert not os.path.exists(REPO_SAVE_DIR), f'{REPO_SAVE_DIR} already exists'
    # create playground
    os.makedirs(REPO_SAVE_DIR)

    logging.info(f'Rank = {rank}, set CURRENT_ISSUE_ID = {CURRENT_ISSUE_ID}')


def reset_current_issue():
    global CURRENT_ISSUE_ID, CURRENT_INSTANCE, CURRENT_STRUCTURE, FOUND_MODULES
    CURRENT_ISSUE_ID = None
    CURRENT_INSTANCE = None
    CURRENT_STRUCTURE = None
    FOUND_MODULES = []

    global ALL_FILE, ALL_CLASS, ALL_FUNC
    ALL_FILE, ALL_CLASS, ALL_FUNC = None, None, None

    global REPO_SAVE_DIR
    if REPO_SAVE_DIR:
        subprocess.run(['rm', '-rf', REPO_SAVE_DIR], check=True)
        REPO_SAVE_DIR = None


def get_current_issue_id():
    global CURRENT_ISSUE_ID
    return CURRENT_ISSUE_ID


def get_current_issue_structure():
    global CURRENT_STRUCTURE
    return CURRENT_STRUCTURE


def get_current_repo_modules():
    global ALL_FILE, ALL_CLASS, ALL_FUNC
    return ALL_FILE, ALL_CLASS, ALL_FUNC


def get_current_issue_data():
    global CURRENT_ISSUE_ID, CURRENT_INSTANCE, CURRENT_STRUCTURE
    return CURRENT_ISSUE_ID, CURRENT_INSTANCE, CURRENT_STRUCTURE


def get_repo_save_dir():
    global REPO_SAVE_DIR
    return REPO_SAVE_DIR


set_current_issue('astropy__astropy-12907')


file_content_in_block_template = """
file: {file_name}
```
{content}
```
"""


def get_repo_structure() -> str:
    """Get the structure of the repository where the issue resides, which you can use to understand the repo and then search related files.

    Args:
        None

    Returns:
        str: The tree structure of the repository where the issue resides.
    """

    structure = get_current_issue_structure()

    # only structure
    structure_str = show_project_structure(structure).strip()
    return structure_str


# TODO:
# sub-structure
def get_directory_structure(dir_path: str) -> str:
    """Search and display the directory structure of a given repository, starting from a specified directory path.

    Args:
        dir_path (str): Directory path to search, formatted as a forward-slash (/) separated string.
    Returns:
        str: The structure of the directory specified in `dir_path`.
    """
    structure = get_current_issue_structure()
    message, current_dir = '', ''

    def show_directory_str(dn, spacing=0):
        return ' ' * spacing + str(dn) + '/' + '\n'

    path = dir_path.split('/')
    path = [p.strip() for p in path if p.strip()]
    if not path:
        structure_str = show_project_structure(structure).strip()
        message += f"Invalid directory path '{dir_path}'.\n"
        message += 'Show the structure of the whole repository.\n'
        return message + structure_str

    s = deepcopy(structure)
    for i, p in enumerate(path):
        if p in s:
            # spacing = i*4
            current_dir = '/'.join(path[: i + 1]) + '/' + '\n'
            s = s[p]
        else:
            if i == 0:
                message += f"No directory named '{p}' in the root.\n"
                message += 'Show the structure of the whole repository.\n'
            else:
                message += f"No directory named '{p}' under '{current_dir}'.\n"
                message += f"Show the structure under '{current_dir}'.\n"
            break

    if i == 0:
        sub_structure_str = show_project_structure(s).strip()
    else:
        sub_structure_str = show_project_structure(s, spacing=4)

    structure_str = message + current_dir + sub_structure_str
    return structure_str


def is_valid_file(file_name: str):
    files, _, _ = get_current_repo_modules()

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, '**/test*/**')
    valid_file_paths = list(set(all_file_paths) - set(exclude_files))
    if file_name in valid_file_paths:
        return True
    else:
        return False


def is_test(name, test_phrases=None):
    if test_phrases is None:
        test_phrases = ['test', 'tests', 'testing']
    words = set(re.split(r' |_|\/|\.', name.lower()))
    return any(word in words for word in test_phrases)


def is_legal_variable_name(name):
    # Regex pattern for a valid Python identifier (variable name)
    valid_variable_pattern = re.compile(r'^[_a-zA-Z][_a-zA-Z0-9]*$')
    return valid_variable_pattern.match(name)


def get_file_content_(file_name: str, return_str=False):
    files, _, _ = get_current_repo_modules()

    for file_content in files:
        if file_content[0] == file_name:
            if return_str:
                content = '\n'.join(file_content[1])
                return content
            else:
                return file_content[1]
    return None


def get_file_content(file_path: str) -> str:
    """Get the entire content of the given file.

    Args:
        file_path: str: The selected file (path) which might be related to this issue.

    Returns:
        str: A string containing the entire content of the corresponding file.

    """

    if not is_valid_file(file_path):
        return 'Invalid file path.'

    file_content = get_file_content_(file_path, return_str=True)
    search_result_str = file_content_in_block_template.format(
        file_name=file_path, content=file_content
    )
    return search_result_str


def get_file_structures(file_list: list[str]) -> str:
    """Get the skeleton/structure of the given files, which you can get all the classes and functions in these files.

    Args:
        file_list: list[str]: The selected files (paths) which might be related to this issue, and the length of this array is not more than 5.

    Returns:
        str: The skeleton/structure of the given files.
    """
    # if isinstance(file_list, str):
    #     file_list = [file_list]
    # if not isinstance(file_list, list):
    #     return None

    search_results = []
    for file_name in file_list:
        if not is_valid_file(file_name):
            search_results.append(f'Invalid file path: {file_name}.')
            continue

        file_content = get_file_content_(file_name, return_str=True)
        file_skeleton = get_skeleton(file_content)
        content = file_content_in_block_template.format(
            file_name=file_name, content=file_skeleton
        )
        search_results.append(content)

    search_result_str = '\n'.join(search_results)
    return search_result_str


def search_class_structures(
    class_names: list[str], file_pattern: Optional[str] = None
) -> str:
    """Retrieves class definitions (skeletons) from the codebase based on the provided class names, filtered by an optional file pattern.

    Args:
        class_names (list[str]): List of class names to search for.
        file_pattern (Optional[str]): A glob pattern to filter the files to search in. Defaults to None, meaning all files are searched.

    Returns:
        str: A formatted string containing file paths, class names, and the method signatures for each class.
    """
    class_content_in_block_template = """
Found class `{cls_name}` in file `{file_name}`:
```
{content}
```
"""
    # if isinstance(class_names, str):
    #     class_names = re.split(r'[\s,]+', class_names)

    # issue_id, bench_data, structure = get_current_issue_data()
    files, classes, _ = get_current_repo_modules()

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, '**/test*/**')
    include_files = all_file_paths
    if file_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_pattern)
    if not include_files:
        file_pattern = None
        include_files = all_file_paths

    all_search_results = []
    for class_name in class_names:
        matched_cls = [cls for cls in classes if class_name == cls['name']]
        if not matched_cls:
            matched_cls = [
                cls
                for cls in classes
                if class_name in cls['name'] or cls['name'] in class_name
            ]

        filter_matched_cls = filter_class(matched_cls, include_files, exclude_files)
        if file_pattern and not filter_matched_cls:
            filter_matched_cls = filter_class(
                matched_cls, all_file_paths, exclude_files
            )

        search_results = []
        for cls in filter_matched_cls:
            # file = cls['file']
            # class_contents[file] = dict()
            content = get_file_content_(cls['file'])
            # class_contents[file][cls['name']] = "\n".join(content[cls['start_line']-1 : cls['end_line']])
            class_content = '\n'.join(content[cls['start_line'] - 1 : cls['end_line']])
            search_result = class_content_in_block_template.format(
                cls_name=cls['name'],
                file_name=cls['file'],
                content=get_skeleton(class_content),
            )
            search_results.append(search_result)

        if search_results:
            all_search_results.append('\n'.join(search_results))
        else:
            all_search_results.append(f'Found no results for class `{class_name}`.')

    class_strucures = '\n'.join(all_search_results)
    return class_strucures


def search_class_contents(class_name: str, exact_match: bool = False) -> dict:
    files, classes, _ = get_current_repo_modules()

    matched_cls = [cls for cls in classes if class_name == cls['name']]
    if not exact_match and not matched_cls:
        matched_cls = [
            cls
            for cls in classes
            if class_name in cls['name'] or cls['name'] in class_name
        ]

    class_contents: Dict[str, dict] = dict()
    for cls in matched_cls:
        file = cls['file']
        class_contents[file] = dict()
        content = get_file_content_(cls['file'])
        class_contents[file][cls['name']] = '\n'.join(
            content[cls['start_line'] - 1 : cls['end_line']]
        )

    return class_contents


def filter_class(class_data: list, include_files: list, exclude_files: list):
    filtered_cls = []
    for cls in class_data:
        file = cls['file']
        if file not in exclude_files and file in include_files:
            filtered_cls.append(cls)
    return filtered_cls


def search_class(class_name: str, file_pattern: Optional[str] = None) -> str:
    """Searching for specified class name within the codebase and retrieves their implementation. This function is essential for quickly locating class implementations, understanding their structures, and analyzing how they fit into the overall architecture of the project.

    Args:
        class_names: str: The class name to search for in the codebase. Please search one class name at a time.
        file_pattern: Optional[str]: A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.

    Returns:
        str: A formatted string containing the search results for specified class, including code snippets of their definitions.
    """

    files, classes, _ = get_current_repo_modules()

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, '**/test*/**')
    include_files = all_file_paths
    if file_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_pattern)
    if not include_files:
        file_pattern = None
        include_files = all_file_paths

    matched_cls = [cls for cls in classes if class_name == cls['name']]
    if not matched_cls:
        matched_cls = [
            cls
            for cls in classes
            if class_name in cls['name'] or cls['name'] in class_name
        ]

    filter_matched_cls = filter_class(matched_cls, include_files, exclude_files)
    if file_pattern and not filter_matched_cls:
        filter_matched_cls = filter_class(matched_cls, all_file_paths, exclude_files)

    all_search_results = []
    # class_contents = dict()
    for cls in filter_matched_cls:
        content = get_file_content_(cls['file'], return_str=True)
        if not content:
            continue
        cls_content = line_wrap_content(content, [(cls['start_line'], cls['end_line'])])
        file_name, cls_name = cls['file'], cls['name']

        search_result = f'Found class `{cls_name}` in `{file_name}`:\n'
        search_result += f'\n{cls_content}\n\n'

        # add dependencies
        dependencies_data = search_invoke_and_reference(
            cls_name, file_name, ntype='class'
        )
        if dependencies_data:
            search_result += dependencies_data
        all_search_results.append(search_result)

    class_contents = '\n'.join(all_search_results)
    return class_contents


def search_function(
    func_name: str,
    all_functions: list[dict],
    exact_match=True,
    include_files: Optional[list[str]] = None,
):
    matched_funcs = [func for func in all_functions if func['name'] == func_name]
    if not exact_match and not matched_funcs:
        matched_funcs = [
            func
            for func in all_functions
            if func['name'] in func_name or func_name in func['name']
        ]

    filtered_matched_funcs = []
    for func in matched_funcs:
        file = func['file']
        if include_files and file in include_files:
            filtered_matched_funcs.append(func)
    return filtered_matched_funcs


def get_method_content(method, class_name, file_name, file_content: str):
    search_result = (
        'Found method `{class_name}.{method_name}` in file `{file_name}`:\n'.format(
            class_name=class_name, method_name=method['name'], file_name=file_name
        )
    )
    method_content = line_wrap_content(
        file_content, [(method['start_line'], method['end_line'])]
    )
    search_result += f'\n{method_content}\n'
    return search_result


def search_method(
    method_name: str,
    class_name: Optional[str] = None,
    file_pattern: Optional[str] = None,
) -> str:
    """Search for specified method name within the codebase and retrieves its definitions along with relevant context. If a class name is provided, the search is limited to methods within that class. This function is essential for quickly locating method implementations, understanding their behaviors, and analyzing their roles in the codebase.

    Args:
        method_name (str): The method (function) name to search for in the codebase.
        class_name (Optional[str]): The name of the class to limit the search scope to methods within this class. If None, the search includes methods in all classes and global functions.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.

    Returns:
        str: A formatted string containing the search result for the specified method, including messages and code snippets of its definition and context.
    """
    # issue_id, bench_data, structure = get_current_issue_data()
    files, classes, functions = get_current_repo_modules()

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, '**/test*/**')
    all_valid_file_paths = list(set(all_file_paths) - set(exclude_files))
    if file_pattern:
        include_files = find_matching_files_from_list(
            all_valid_file_paths, file_pattern
        )
        if not include_files:
            file_pattern = None
            include_files = all_valid_file_paths
    else:
        include_files = all_valid_file_paths

    init_method_name = method_name
    if '.' in method_name:
        method_name = method_name.split('.')[-1]

    search_results = []
    found_method = False

    if class_name:
        add_dependency = False
        # Exact Match first
        cls_names = re.split(r'\s+|\s*\.\s*', class_name)
        cls_names = [
            item for item in cls_names if item and is_legal_variable_name(item)
        ]

        matched_cls = [
            cls
            for cls in classes
            if cls['name'] in cls_names and cls['file'] in include_files
        ]
        if not matched_cls and file_pattern:
            # ignore file_pattern
            matched_cls = [
                cls
                for cls in classes
                if cls['name'] in cls_names and cls['file'] in all_valid_file_paths
            ]

        if matched_cls:
            add_dependency = True
        else:
            search_result = f'Found no class named `{class_name}`.'
            search_results.append(search_result)
            # fuzzy search for class
            matched_cls = [
                cls
                for cls in classes
                if (cls['name'] in class_name or class_name in cls['name'])
                and cls['file'] in include_files
            ]

        if not matched_cls and file_pattern:
            matched_cls = [
                cls
                for cls in classes
                if (cls['name'] in class_name or class_name in cls['name'])
                and cls['file'] in all_valid_file_paths
            ]

        for cls in matched_cls:
            matched_methods = [
                method for method in cls['methods'] if method['name'] == method_name
            ]
            if matched_methods:
                found_method = True
                file_content = get_file_content_(cls['file'], return_str=True)
            for method in matched_methods:
                search_result = get_method_content(
                    method, cls['name'], cls['file'], file_content
                )
                method_sig = '{class_name}.{method_name}'.format(
                    class_name=cls['name'], method_name=method['name']
                )

                if add_dependency:
                    # add dependencies
                    dependencies_data = search_invoke_and_reference(
                        method_sig, cls['file'], ntype='function'
                    )
                    if dependencies_data:
                        search_result += dependencies_data

                search_results.append(search_result)

        # search any class which has such a method
        if not found_method:
            for cls in classes:
                matched_methods = [
                    method for method in cls['methods'] if method['name'] == method_name
                ]
                if not matched_methods:
                    continue
                if matched_methods:
                    found_method = True
                    file_content = get_file_content_(cls['file'], return_str=True)
                for method in matched_methods:
                    search_result = get_method_content(
                        method, cls['name'], cls['file'], file_content
                    )
                    search_results.append(search_result)

        if not found_method:
            search_result = (
                f'Found no results for method `{init_method_name}` in any class.'
            )
            search_results.append(search_result)

    if not found_method:
        # search function directly
        add_dependency = False
        # func_names = [func for func in method_name.split() if is_legal_variable_name(func)]

        # Exact match first
        filtered_matched_funcs = search_function(
            method_name, functions, exclude_files, include_files
        )
        if not filtered_matched_funcs and file_pattern:
            filtered_matched_funcs = search_function(
                method_name, functions, all_valid_file_paths
            )

        if filtered_matched_funcs:
            add_dependency = True
        else:
            # fuzzy search
            filtered_matched_funcs = search_function(
                method_name, functions, False, include_files
            )
        if not filtered_matched_funcs and file_pattern:
            filtered_matched_funcs = search_function(
                method_name,
                functions,
                exact_match=False,
                include_files=all_valid_file_paths,
            )

        for func in filtered_matched_funcs:
            found_method = True
            content = get_file_content_(func['file'], return_str=True)
            if not content:
                continue
            search_result = 'Found method `{method_name}` in file `{file}`:\n'.format(
                method_name=func['name'], file=func['file']
            )
            func_content = line_wrap_content(
                content, [(func['start_line'], func['end_line'])]
            )
            search_result += f'\n{func_content}\n\n'

            if add_dependency:
                # add dependencies
                dependencies_data = search_invoke_and_reference(
                    func['name'], func['file'], ntype='function'
                )
                if dependencies_data:
                    search_result += dependencies_data
                search_results.append(search_result)

    search_result_str = ''
    if search_results:
        search_result_str = '\n'.join(search_results)

    if not found_method:
        search_result_str += f'Found no results for method {init_method_name}.\n'
        if class_name:
            search_result_str += f'Searching for class `{class_name}` ...\n'
            search_result_str += search_class_structures([class_name], file_pattern)

    return search_result_str


def search_in_repo(
    search_terms: list[str], file_pattern: Optional[str] = '**/*.py'
) -> str:
    """Performs a combined search using both the BM25 algorithm and semantic search on the codebase.
    For each search term, this function retrieves code snippets by first performing a BM25 search to rank documents based on
    the similarity to the term and then follows up with a semantic search to find more contextually similar code snippets.

    Args:
        search_terms (list[str]): Textual queries used to search the codebase. Each can be a functional description, a potential class or method name, or any relevant terms related to the code you want to find in the repository.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.

    Returns:
        str: A formatted string containing the combined results from both BM25 and semantic search.
            Each search result contains the file path and the retrieved code snippet (the partial code of a module or just the skeleton of the specific module).
    """
    # issue_id, instance, structure = get_current_issue_data()
    files, classes, functions = get_current_repo_modules()
    # all_file_paths = [file[0] for file in files]
    all_class_names = [cls['name'] for cls in classes]
    all_function_names = [func['name'] for func in functions]

    result = ''
    for term in search_terms:
        result += f"Searching for '{term}' ...\n\n## Searching Result:\n"
        cur_result = ''
        if term in all_class_names:
            cur_result += search_class(term, file_pattern)
        if term in all_function_names:
            cur_result += search_method(term, file_pattern)
            if 'Found no results' in cur_result:
                cur_result = ''
        if '.' in term and len(term.split('.')) == 2:
            cur_result += search_method(
                term.split('.')[-1], term.split('.')[0], file_pattern
            )
            if 'Found no results' in cur_result:
                cur_result = ''

        if cur_result:
            result += cur_result
        else:
            result += search_term_in_repo(term, file_pattern)
    return result


def search_term_in_repo(query: str, file_pattern: Optional[str] = '**/*.py') -> str:
    """Performs a combined search using both the BM25 algorithm and semantic search on the codebase.
    This function retrieves code snippets by first performing a BM25 search to rank documents based on
    the similarity to the query and then follows up with a semantic search to find more contextually
    similar code snippets.

    Args:
        query (str): A textual query used to search the codebase. This can be a functional description, a potential class or method name, or any relevant terms related to the code you want to find in the repository.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.

    Returns:
        str: A formatted string containing the combined results from both BM25 and semantic search, including file paths and the retrieved code snippets (the partial code of a module or the skeleton of the specific module).
    """
    bm25_result = '### Retrieving by bm25 algorithm:\n'
    bm25_result += bm25_retrieve(
        query=query, file_pattern=file_pattern, similarity_top_k=10
    )

    semantic_result = '### Retrieving by semantic search:\n'
    semantic_result += semantic_search(
        query=query, file_pattern=file_pattern, max_results=10, use_skeleton=False
    )

    retrieve_result = bm25_result + '\n' + semantic_result
    return retrieve_result


def bm25_retrieve(
    query: str, file_pattern: Optional[str] = None, similarity_top_k: int = 15
) -> str:
    """Retrieves code snippets from the codebase using the BM25 algorithm based on the provided query, class names, and function names. This function helps in finding relevant code sections that match specific criteria, aiding in code analysis and understanding.

    Args:
        query (Optional[str]): A textual query to search for relevant code snippets. Defaults to an empty string if not provided.
        class_names (list[str]): A list of class names to include in the search query. If None, class names are not included.
        function_names (list[str]): A list of function names to include in the search query. If None, function names are not included.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.
        similarity_top_k (int): The number of top similar documents to retrieve based on the BM25 ranking. Defaults to 15.

    Returns:
        str: A formatted string containing the search results, including file paths and the retrieved code snippets (the partial code of a module or the skeleton of the specific module).
    """

    issue_id, instance, structure = get_current_issue_data()
    files, _, _ = get_current_repo_modules()

    repo_playground = get_repo_save_dir()
    repo_dir = setup_full_swebench_repo(
        instance_data=instance, repo_base_dir=repo_playground
    )
    persist_dir = os.path.join(
        INDEX_STORE_LOC, get_repo_dir_name(instance['instance_id'])
    )
    workspace = Workspace.from_dirs(repo_dir=repo_dir, index_dir=persist_dir)
    code_index = workspace.code_index
    # bm25_retriever = BM25Retriever.from_persist_dir("plugins/retrieval_tools/bm25_retriever/persist_data/")
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=code_index._docstore,
        similarity_top_k=10,
        stemmer=Stemmer.Stemmer('english'),
        language='english',
    )

    # message = f"Search results for query `{query}`:\n"
    message = ''

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, '**/test*/**')
    if file_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_pattern)
        if not include_files:
            include_files = all_file_paths
            message += f'No files found for file pattern {file_pattern}. Will search all files.\n...\n'
    else:
        include_files = all_file_paths

    retrieved_nodes = bm25_retriever.retrieve(query)
    result_template = """file: {file}
```
{code_content}
```
"""
    # similarity: {score}

    search_result_strs = []
    for node in retrieved_nodes:
        file = node.metadata['file_path']
        if file not in exclude_files and file in include_files:
            if (
                len(node.metadata['span_ids']) == 1
                and node.metadata['span_ids'][0] == 'imports'
            ):
                continue
            content = get_file_content_(file, return_str=True)
            if not content:
                continue
            result_content = line_wrap_content(
                content, [(node.metadata['start_line'], node.metadata['end_line'])]
            )
            search_result_str = result_template.format(
                file=file, code_content=result_content
            )

            # code_content = node.node.get_content().strip()
            # if node.metadata['tokens'] <= 100:
            #     search_result_str = result_template.format(
            #         file=file,
            #         # score=node.score,
            #         code_content=code_content
            #     )
            # else:
            #     code_structure = get_skeleton(code_content).strip()
            #     search_result_str = result_template.format(
            #         file=file,
            #         # score=node.score,
            #         code_content=code_structure
            #     )
            search_result_strs.append(search_result_str)
    if search_result_strs:
        search_result_strs = search_result_strs[:5]  # 5 at most
        # message += f'Found {len(search_result_strs)} code spans. The skeleton of each module are as follows.\n\n'
        message += f'Found {len(search_result_strs)} code spans.\n\n'
        return message + '\n'.join(search_result_strs)
    else:
        return 'No locations found.'


def extract_file_to_code(raw_content: str):
    # import re
    # Pattern to extract the file name and code
    pattern = r'([\w\/\.]+)\n```\n(.*?)\n```'

    # Use re.findall to extract all matches (file name and code)
    matches = re.findall(pattern, raw_content, re.DOTALL)

    # Create a dictionary from the extracted file names and code
    file_to_code = {filename: code for filename, code in matches}

    return file_to_code


def semantic_search(
    query: Optional[str] = None,
    code_snippet: Optional[str] = None,
    class_names: Optional[list[str]] = None,
    function_names: Optional[list[str]] = None,
    use_skeleton: bool = False,
    file_pattern: Optional[str] = '**/*.py',
    max_results: int = 10,
) -> str:
    """Performs a semantic search over the codebase by incorporating the provided query, code snippet, class names, and function names into a combined search query. This function includes the specified class names and function names in the query before performing the semantic search, enabling a more targeted and relevant search for semantically similar code snippets.

    Args:
        query (Optional[str]): A textual query describing what to search for in the codebase.
        code_snippet (Optional[str]): A code snippet to find semantically similar code in the codebase.
        class_names (list[str]): A list of class names to include in the search query.
        function_names (list[str]): A list of function names to include in the search query.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.

    Returns:
        str: A formatted string containing the search results, including messages and code snippets of the semantically similar code sections found.
    """

    issue_id, instance, structure = get_current_issue_data()
    repo_playground = get_repo_save_dir()
    repo_dir = setup_full_swebench_repo(
        instance_data=instance, repo_base_dir=repo_playground
    )
    persist_dir = os.path.join(
        INDEX_STORE_LOC, get_repo_dir_name(instance['instance_id'])
    )
    workspace = Workspace.from_dirs(repo_dir=repo_dir, index_dir=persist_dir)
    # file_context = workspace.create_file_context()
    file_context = workspace.file_context

    # message = f"Searching for query [{query[:20]}...] and file pattern [{file_pattern}]."
    message = ''
    search_response = workspace.code_index.semantic_search(
        query=query,
        code_snippet=code_snippet,
        class_names=class_names,
        function_names=function_names,
        file_pattern=file_pattern,
        max_results=max_results,
    )
    if not search_response.hits:
        message += f'No files found for file pattern {file_pattern}. Will search all files.\n...\n'
        search_response = workspace.code_index.semantic_search(
            query=query,
            code_snippet=code_snippet,
            class_names=class_names,
            function_names=function_names,
            file_pattern=None,
            max_results=max_results,
        )

    for hit in search_response.hits:
        for span in hit.spans:
            file_context.add_span_to_context(
                hit.file_path,
                span.span_id,
                tokens=1,  # span.tokens
            )
    if file_context.files:
        file_context.expand_context_with_init_spans()
    result_template = """file: {file}
```
...
{code_content}

...
```
"""
    if use_skeleton:
        file_to_search_result = extract_file_to_code(
            file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=False,
                exclude_comments=False,
            )
        )
        search_result_str = f'Found {len(file_to_search_result)} code spans. The skeleton of each module are as follows.\n\n'
        file_to_skeleteons = dict()
        for file, code_content in file_to_search_result.items():
            code_structure = get_skeleton(code_content).strip()
            file_to_skeleteons[file] = code_structure
        search_result_str += '\n'.join(
            [
                result_template.format(file=file, code_content=code)
                for file, code in file_to_skeleteons.items()
            ]
        )
    else:
        search_result_str = search_response.message + '\n'
        search_result_str += file_context.create_prompt(exclude_comments=False)

    return search_result_str


def search_dependency_graph_one_hop(
    issue_id: str, module_name: str, file_path: str, ntype: str
):
    if ntype == 'file':
        nid = file_path
    elif ntype == 'function' or ntype == 'class':
        nid = file_path + ':' + module_name
    else:
        raise NotImplementedError

    G = pickle.load(open(f'{DEPENDENCY_GRAPH_LOC}/{issue_id}.pkl', 'rb'))

    if nid not in G:
        return None

    searcher = RepoSearcher(G)
    return searcher.one_hop_neighbors(nid, return_data=True)


def search_invoke_and_reference(module_name: str, file_path: str, ntype: str) -> str:
    """Analyzes the dependencies of a specific class, or function, identifying how they interact with other parts of the codebase. This is useful for understanding the broader impact of any changes or issues and ensuring that all related components are accounted for.

    Args:
        module_name (str): The name of the module to analyze. This could refer to the name of a class or a function within the codebase. Example values: "class_A", "function_1" or "class_B.function_2".
        file_path (str): The full path to the file where the module resides. This helps in precisely locating the module within the codebase.
        ntype (str): The type of the module being analyzed. Must be one of the following values:
                    - 'class': when analyzing a class's dependencies.
                    - 'function': when analyzing a function's dependencies.

    Returns:
        str: A string containing the dependencies for the specified module, which shows other modules that invoke
             or are invoked by this module. Returns `None` if no dependencies are found.
    Raises:
        NotImplementedError: If the `ntype` is not one of 'class', or 'function'.

    Example:
        search_invoke_and_reference("function_1", "path/to/file1.py", "function")
    """
    graph_context = ''
    graph_item_format = """
### Dependencies for {ntype} `{module}` in `{fname}`:
{dependencies}
"""
    tag_format = """
location: {fname} lines {start_line} - {end_line}
name: {name}
contents:
{contents}

"""

    issue_id = get_current_issue_id()
    files, classes, functions = get_current_repo_modules()

    # 判断参数的有效性
    if ntype == 'class':
        selected_class = [
            cls
            for cls in classes
            if cls['name'] == module_name and cls['file'] == file_path
        ]
        if not selected_class:
            selected_class = [cls for cls in classes if cls['name'] == module_name]
            if len(selected_class) == 1:
                file_path = selected_class[0]['file']
            else:
                return ''
    elif ntype == 'function' and '.' in module_name:
        class_name = module_name.split('.')[0]
        method_name = module_name.split('.')[-1]

        selected_class = [
            cls
            for cls in classes
            if cls['name'] == class_name and cls['file'] == file_path
        ]
        if selected_class:
            selected_class_ = selected_class[0]
            selected_method = [
                method
                for method in selected_class_['methods']
                if method['name'] == method_name
            ]
            if not selected_method:
                return ''
        else:
            selected_class = [cls for cls in classes if cls['name'] == module_name]
            if len(selected_class) == 1:
                selected_class_ = selected_class[0]
                selected_method = [
                    method
                    for method in selected_class_['methods']
                    if method['name'] == method_name
                ]
                if selected_method:
                    file_path = selected_class_['file']
                else:
                    return ''
            else:
                return ''
    else:
        selected_func = [
            func
            for func in functions
            if func['name'] == module_name and func['file'] == file_path
        ]
        if not selected_func:
            selected_func = [func for func in functions if func['name'] == module_name]
            if len(selected_func) == 1:
                file_path = selected_func[0]['file']
            else:
                return ''

    results = search_dependency_graph_one_hop(issue_id, module_name, file_path, ntype)

    if not results:
        return ''
    filter_results = []
    for result in results:
        if result['type'] == 'file' or is_test(result['file_path']):
            continue
        if 'contains' in result['relation']:
            continue
        filter_results.append(result)

    if not filter_results:
        return ''

    code_graph_context = ''
    for result in filter_results:
        f_name = result['file_path']
        m_name = result['module_name']
        content = get_file_content_(f_name, return_str=True)
        if not content:
            continue

        if result['type'] == 'class':
            selected_class = [
                cls
                for cls in classes
                if cls['name'] == result['module_name'] and cls['file'] == f_name
            ]
            selected_class_ = selected_class[0]
            try:
                init_method = [
                    method
                    for method in selected_class_['methods']
                    if method['name'] == '__init__'
                ][0]
                m_name = f'{m_name}.__init__'
                start_line = selected_class_['start_line']
                end_line = init_method['end_line']
            except Exception as e:
                print(f'An error occurred: {e}')  # TODO: logger
                start_line = selected_class_['start_line']
                end_line = selected_class_['end_line']
        else:
            start_line = result['start_line']
            end_line = result['end_line']

        # module_content = content[result['start_line']-1: result['end_line']]
        module_content = line_wrap_content(content, [(start_line, end_line)])
        code_graph_context += tag_format.format(
            fname=f_name,
            name=m_name,
            start_line=start_line,
            end_line=end_line,
            contents=module_content,
        )
    graph_context += graph_item_format.format(
        ntype=ntype,
        module=module_name,
        fname=file_path,
        dependencies=code_graph_context,
    )
    return graph_context


def extract_module_id(text: str):
    # Regular expression pattern to extract the file_name, ntype, and module_name
    pattern = r'(?P<file_name>[\w/]+\.py)\s\((?P<ntype>\w+):\s(?P<module_name>[\w.]+)\)'

    # Use re.search to match the pattern
    match = re.search(pattern, text)

    if match:
        file_name = match.group('file_name')
        module_name = match.group('module_name')
        ntype = match.group('ntype')
        module_id = f'{file_name}:{module_name}'
        return (module_id, ntype)
    else:
        return (None, None)


def get_formatted_node_str(nid, nodes_data):
    if ':' in nid:
        file_name, module_name = nid.split(':')
        for node in nodes_data:
            if node['file_path'] == file_name and node['module_name'] == module_name:
                break
        ntype = node['type']
        formatted_text = f'{file_name} ({ntype}: {module_name})'
    else:
        formatted_text = nid

    return formatted_text


def search_interactions_among_modules(module_ids: list[str]):
    """Analyze the interactions between specified modules by examining a pre-built dependency graph.
    Args:
        module_ids (list[str]): A list of unique identifiers for modules to analyze.
                                Each identifier corresponds to either a file or a function/class within a file:
                                - For files, the identifier is the full file path, e.g., 'full_path1/file1.py'.
                                - For functions or classes, the identifier includes the file path and the module name, e.g.,
                                  'full_path1/file1.py (function: MyClass1.entry_function)' or 'full_path1/file1.py (class: MyClass2)'.
    Returns:
        str: A formatted string describing the interactions (edges) between the specified modules in the dependency graph.
             The format shows relationships between source and target modules, such as:
             'source_file (type: source_module) -> relation -> target_file (type: target_module)'.
             Returns `None` if no interactions are found.
    """

    issue_id, _, structure = get_current_issue_data()
    G = pickle.load(open(f'{DEPENDENCY_GRAPH_LOC}/{issue_id}.pkl', 'rb'))
    searcher = RepoSearcher(G)

    nids = []
    for mid in module_ids:
        if '(' not in mid and mid in G:
            nids.append(mid)
        else:
            module_id, ntype = extract_module_id(mid)
            if module_id and module_id in G:
                nids.append(module_id)

    if not nids:
        return 'None'

    # Initialize an empty set to store all the nodes in the paths
    nodes_in_paths = set()

    # Collect all nodes and edges that are part of the paths between pairs of nodes
    for i, node in enumerate(nids):
        for other_node in nids[i + 1 :]:
            try:
                # Find the shortest path between the pair
                path = nx.shortest_path(G, source=node, target=other_node)
                nodes_in_paths.update(path)
            except nx.NetworkXNoPath:
                continue

    if nodes_in_paths:
        print(nodes_in_paths)
        edges, node_data = searcher.subgraph(nodes_in_paths)
    else:
        edges, node_data = searcher.subgraph(nids)
    # print(nids, edges)

    edge_str = ''
    edge_template = """{source_nid} -> {relation} -> {tartget_id}\n"""
    for edge in edges:
        source_nid = get_formatted_node_str(edge[0], node_data)
        tartget_id = get_formatted_node_str(edge[1], node_data)

        edge_str += edge_template.format(
            source_nid=source_nid, relation=edge[2], tartget_id=tartget_id
        )
    return edge_str


__all__ = [
    'get_repo_structure',
    'get_directory_structure',
    'get_file_content',
    'get_file_structures',
    'search_invoke_and_reference',
    'search_class',
    'search_class_structures',
    'search_method',
    'search_in_repo',
    # 'search_interactions_among_modules'
]
