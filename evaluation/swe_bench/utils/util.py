import fnmatch
import json
import logging
import os
import zipfile
from collections import defaultdict
from copy import deepcopy
from typing import DefaultDict, List, Optional

from datasets import load_dataset

from evaluation.swe_bench.utils.repo import setup_github_repo

# SET THIS IF YOU WANT TO USE THE PREPROCESSED FILES
PROJECT_FILE_LOC = os.environ.get('PROJECT_FILE_LOC')
# DEPENDENCY_GRAPH_LOC = os.environ.get("DEPENDENCY_GRAPH_LOC")
# INDEX_STORE_LOC = os.environ.get("INDEX_STORE_LOC")

def filter_none_python(structure):
    for key, value in list(structure.items()):
        if (
            'functions' not in value.keys()
            and 'classes' not in value.keys()
            and 'text' not in value.keys()
        ) or not len(value.keys()) == 3:
            filter_none_python(value)

            if structure[key] == {}:
                del structure[key]
        else:
            if not key.endswith('.py'):
                del structure[key]
                
                
def filter_out_test_files(structure):
    """filter out test files from the project structure"""
    for key, value in list(structure.items()):
        if key.startswith('test'):
            del structure[key]
        elif isinstance(value, dict):
            filter_out_test_files(value)

   
def find_matching_files_from_list(file_list, file_pattern):
    """
    Find and return a list of file paths from the given list that match the given keyword or pattern.

    :param file_list: A list of file paths to search through.
    :param file_pattern: A keyword or pattern for file matching. Can be a simple keyword or a glob-style pattern.
    :return: A list of matching file paths
    """
    # If the pattern contains any of these glob-like characters, treat it as a glob pattern.
    if '*' in file_pattern or '?' in file_pattern or '[' in file_pattern:
        matching_files = fnmatch.filter(file_list, file_pattern)
    else:
        # Otherwise, treat it as a keyword search
        matching_files = [file for file in file_list if file_pattern in file]

    return matching_files


def get_meta_data(
    target_id, dataset: str = 'princeton-nlp/SWE-bench_Lite', split: str = 'test'
):
    # swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    swe_bench_data = load_dataset(dataset, split=split)
    bench_data = [x for x in swe_bench_data if x['instance_id'] == target_id][0]
    return bench_data


def get_repo_structures(bench_data):
    if PROJECT_FILE_LOC is not None:
        project_file = os.path.join(
            PROJECT_FILE_LOC, bench_data['instance_id'] + '.json'
        )
        d = load_json(project_file)
    else:
        logging.info(
            '`PROJECT_FILE_LOC` is None, get the project structure from scratch'
        )
        # we need to get the project structure directly
        # d = get_project_structure_from_scratch(
        #     bench_data['repo'],
        #     bench_data['base_commit'],
        #     bench_data['instance_id'],
        #     'playground',
        # )

    # instance_id = d['instance_id']
    structure = d['structure']
    filter_none_python(structure)

    # some basic filtering steps
    # filter out test files (unless its pytest)
    if not d['instance_id'].startswith('pytest'):
        filter_out_test_files(structure)

    return structure


def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, 'r') as file:
        return [json.loads(line) for line in file]


def write_jsonl(data, filepath):
    """
    Write data to a JSONL file at the given filepath.

    Arguments:
    data -- a list of dictionaries to write to the JSONL file
    filepath -- the path to the JSONL file to write
    """
    with open(filepath, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')


def load_json(filepath):
    return json.load(open(filepath, 'r'))


def combine_by_instance_id(data):
    """
    Combine data entries by their instance ID.

    Arguments:
    data -- a list of dictionaries with instance IDs and other information

    Returns:
    A list of combined dictionaries by instance ID with all associated data.
    """
    combined_data: DefaultDict[str, DefaultDict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in data:
        instance_id = item.get('instance_id')
        if not instance_id:
            continue
        for key, value in item.items():
            if key != 'instance_id':
                combined_data[instance_id][key].extend(
                    value if isinstance(value, list) else [value]
                )
    return [
        {**{'instance_id': iid}, **details} for iid, details in combined_data.items()
    ]


def get_full_file_paths_and_classes_and_functions(structure, current_path=''):
    """
    Recursively retrieve all file paths, classes, and functions within a directory structure.

    Arguments:
    structure -- a dictionary representing the directory structure
    current_path -- the path accumulated so far, used during recursion (default="")

    Returns:
    A tuple containing:
    - files: list of full file paths
    - classes: list of class details with file paths
    - functions: list of function details with file paths
    """
    files = []
    classes = []
    functions = []
    for name, content in structure.items():
        if isinstance(content, dict):
            if (
                'functions' not in content.keys()
                and 'classes' not in content.keys()
                and 'text' not in content.keys()
            ) or not len(content.keys()) == 3:
                # or guards against case where functions and classes are somehow part of the structure.
                next_path = f'{current_path}/{name}' if current_path else name
                (
                    sub_files,
                    sub_classes,
                    sub_functions,
                ) = get_full_file_paths_and_classes_and_functions(content, next_path)
                files.extend(sub_files)
                classes.extend(sub_classes)
                functions.extend(sub_functions)
            else:
                next_path = f'{current_path}/{name}' if current_path else name
                files.append((next_path, content['text']))
                if 'classes' in content:
                    for clazz in content['classes']:
                        classes.append(
                            {
                                'file': next_path,
                                'name': clazz['name'],
                                'start_line': clazz['start_line'],
                                'end_line': clazz['end_line'],
                                'methods': [
                                    {
                                        'name': method['name'],
                                        'start_line': method['start_line'],
                                        'end_line': method['end_line'],
                                    }
                                    for method in clazz.get('methods', [])
                                ],
                            }
                        )
                if 'functions' in content:
                    for function in content['functions']:
                        function['file'] = next_path
                        functions.append(function)
        else:
            next_path = f'{current_path}/{name}' if current_path else name
            files.append(next_path)
    return files, classes, functions


def load_instances(
    dataset_name: str = 'princeton-nlp/SWE-bench_Lite', split: str = 'test'
):
    data = load_dataset(dataset_name, split=split)
    return {d['instance_id']: d for d in data}


def load_instance(
    instance_id: str,
    dataset_name: str = 'princeton-nlp/SWE-bench_Lite',
    split: str = 'test',
):
    data = load_instances(dataset_name, split=split)
    return data[instance_id]


def setup_swebench_repo(
    instance_data: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert (
        instance_data or instance_id
    ), 'Either instance_data or instance_id must be provided'

    if not instance_data:
        instance_data = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv('REPO_DIR', '/tmp/repos')

    repo_dir_name = instance_data['repo'].replace('/', '__')
    github_repo_path = f'swe-bench/{repo_dir_name}'
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data['base_commit'],
        base_dir=repo_base_dir,
    )


def setup_full_swebench_repo(
    instance_data: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert (
        instance_data or instance_id
    ), 'Either instance_data or instance_id must be provided'
    if not instance_data:
        instance_data = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv('REPO_DIR', '/tmp/repos')

    # repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = instance_data['repo']
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data['base_commit'],
        base_dir=repo_base_dir,
    )


def get_repo_dir_name(repo: str):
    return repo.replace('/', '_')


def zip_directory(directory_path):
    # Get the directory name for the zip file
    zip_file_path = f'{directory_path}.zip'

    # Create a ZipFile object and add the directory to it
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to the zip file with the relative path
                zip_file.write(file_path, os.path.relpath(file_path, directory_path))

    return zip_file_path


def get_all_valid_files(instance_id=None, instance_data=None):
    assert instance_id or instance_data
    if instance_id:
        instance = get_meta_data(instance_id)
    elif instance_data:
        instance = instance_data
    structure = get_repo_structures(instance)
    files, _, _ = get_full_file_paths_and_classes_and_functions(structure)

    all_valid_files = []
    for file_content in files:
        file = file_content[0]
        all_valid_files.append(file)
    return all_valid_files