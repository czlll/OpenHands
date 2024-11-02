import argparse
import ast
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def handle_edge_cases(code):
    # hard-coded edge cases
    code = code.replace('\ufeff', '')
    code = code.replace('constants.False', '_False')
    code = code.replace('constants.True', '_True')
    code = code.replace('False', '_False')
    code = code.replace('True', '_True')
    code = code.replace('DOMAIN\\username', 'DOMAIN\\\\username')
    code = code.replace('Error, ', 'Error as ')
    code = code.replace('Exception, ', 'Exception as ')
    code = code.replace('print ', 'yield ')
    pattern = r'except\s+\(([^,]+)\s+as\s+([^)]+)\):'
    # Replace 'as' with ','
    code = re.sub(pattern, r'except (\1, \2):', code)
    code = code.replace('raise AttributeError as aname', 'raise AttributeError')
    return code


def find_imports(filepath, repo_path):
    # root_path: 项目根目录
    try:
        with open(filepath, 'r') as file:
            tree = ast.parse(file.read(), filename=filepath)
    except Exception:
        raise SyntaxError

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # Handle 'import module' and 'import module as alias'
            for alias in node.names:
                module_name = alias.name
                asname = alias.asname
                imports.append(
                    {'type': 'import', 'module': module_name, 'alias': asname}
                )
        elif isinstance(node, ast.ImportFrom):
            # Handle 'from ... import ...' statements
            import_entities = []
            for alias in node.names:
                if alias.name == '*':
                    import_entities = [{'name': '*', 'alias': None}]
                    break
                else:
                    entity_name = alias.name
                    asname = alias.asname
                    import_entities.append({'name': entity_name, 'alias': asname})

            # Calculate the module name for relative imports
            if node.level == 0:
                # Absolute import
                module_name = node.module if node.module is not None else ''
            else:
                # Relative import
                rel_path = os.path.relpath(filepath, repo_path)
                # rel_dir = os.path.dirname(rel_path)
                package_parts = rel_path.split(os.sep)

                # Adjust for the level of relative import
                if len(package_parts) >= node.level:
                    package_parts = package_parts[: -node.level]
                else:
                    package_parts = []

                if node.module:
                    module_name = '.'.join(package_parts + [node.module])
                else:
                    module_name = '.'.join(package_parts)

            imports.append(
                {'type': 'from', 'module': module_name, 'entities': import_entities}
            )
    return imports


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.nodes = []
        self.node_name_stack = []
        self.node_type_stack = []

    def visit_ClassDef(self, node):
        class_name = node.name
        full_class_name = '.'.join(self.node_name_stack + [class_name])
        self.nodes.append(
            {
                'name': full_class_name,
                'type': 'class',
                'code': self._get_source_segment(node),
                'start_line': node.lineno,
                'end_line': node.end_lineno,
            }
        )

        self.node_name_stack.append(class_name)
        self.node_type_stack.append('class')
        self.generic_visit(node)
        self.node_name_stack.pop()
        self.node_type_stack.pop()

    def visit_FunctionDef(self, node):
        if (
            self.node_type_stack
            and self.node_type_stack[-1] == 'class'
            and node.name == '__init__'
        ):
            return
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node)

    def _visit_func(self, node):
        function_name = node.name
        full_function_name = '.'.join(self.node_name_stack + [function_name])
        self.nodes.append(
            {
                'name': full_function_name,
                'parent_type': self.node_type_stack[-1]
                if self.node_type_stack
                else None,
                'type': 'function',
                'code': self._get_source_segment(node),
                'start_line': node.lineno,
                'end_line': node.end_lineno,
            }
        )

        self.node_name_stack.append(function_name)
        self.node_type_stack.append('function')
        self.generic_visit(node)
        self.node_name_stack.pop()
        self.node_type_stack.pop()

    def _get_source_segment(self, node):
        with open(self.filename, 'r') as file:
            source_code = file.read()
        return ast.get_source_segment(source_code, node)


# 解析指定文件，使用CodeAnalyzer分析文件中的类和顶级函数
def analyze_file(filepath):
    with open(filepath, 'r') as file:
        code = file.read()
        # code = handle_edge_cases(code)
        try:
            tree = ast.parse(code, filename=filepath)
        except Exception:
            raise SyntaxError
    analyzer = CodeAnalyzer(filepath)
    try:
        analyzer.visit(tree)
    except RecursionError:
        pass
    return analyzer.nodes


def resolve_module(module_name, repo_path):
    """
    Resolve a module name to a file path in the repo.
    Returns the file path if found, or None if not found.
    """
    # Try to resolve as a .py file
    module_path = os.path.join(repo_path, module_name.replace('.', '/') + '.py')
    if os.path.isfile(module_path):
        return module_path

    # Try to resolve as a package (__init__.py)
    init_path = os.path.join(repo_path, module_name.replace('.', '/'), '__init__.py')
    if os.path.isfile(init_path):
        return init_path

    return None


# 遍历repo_path下的所有Python文件，构建文件、类和函数的依赖关系图
def build_graph_v2(repo_path, fuzzy_search=True, global_import=False):
    graph = nx.MultiDiGraph()
    file_nodes = {}

    # add nodes
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                try:
                    file_path = os.path.join(root, file)
                    filename = os.path.relpath(file_path, repo_path)
                    with open(file_path, 'r') as f:
                        file_content = f.read()

                    graph.add_node(filename, type='file', code=file_content)
                    file_nodes[filename] = file_path

                    nodes = analyze_file(file_path)
                except (UnicodeDecodeError, SyntaxError):
                    # Skip the file that cannot decode or parse
                    continue

                # add nodes
                for node in nodes:
                    full_name = f'{filename}:{node["name"]}'
                    graph.add_node(
                        full_name,
                        type=node['type'],
                        code=node['code'],
                        start_line=node['start_line'],
                        end_line=node['end_line'],
                    )

                # add edges with type=contains
                for node in nodes:
                    full_name = f'{filename}:{node["name"]}'
                    name_list = node['name'].split('.')
                    if len(name_list) == 1:
                        graph.add_edge(filename, full_name, type='contains')
                    else:
                        parent_name = '.'.join(name_list[:-1])
                        full_parent_name = f'{filename}:{parent_name}'
                        graph.add_edge(full_parent_name, full_name, type='contains')

    # add 'imports' edges
    for filename, filepath in file_nodes.items():
        try:
            imports = find_imports(filepath, repo_path)
        except SyntaxError:
            continue

        for imp in imports:
            if imp['type'] == 'import':
                # Handle 'import module' statements
                module_name = imp['module']
                module_path = resolve_module(module_name, repo_path)
                if module_path:
                    imp_filename = os.path.relpath(module_path, repo_path)
                    if graph.has_node(imp_filename):
                        graph.add_edge(
                            filename, imp_filename, type='imports', alias=imp['alias']
                        )
            elif imp['type'] == 'from':
                # Handle 'from module import entity' statements
                module_name = imp['module']
                entities = imp['entities']

                if len(entities) == 1 and entities[0]['name'] == '*':
                    # Handle 'from module import *' as 'import module' statement
                    module_path = resolve_module(module_name, repo_path)
                    if module_path:
                        imp_filename = os.path.relpath(module_path, repo_path)
                        if graph.has_node(imp_filename):
                            graph.add_edge(
                                filename, imp_filename, type='imports', alias=None
                            )
                    continue  # Skip further processing for 'import *'

                for entity in entities:
                    entity_name, entity_alias = entity['name'], entity['alias']
                    entity_module_name = f'{module_name}.{entity_name}'
                    entity_module_path = resolve_module(entity_module_name, repo_path)
                    if entity_module_path:
                        # Entity is a submodule
                        entity_filename = os.path.relpath(entity_module_path, repo_path)
                        if graph.has_node(entity_filename):
                            graph.add_edge(
                                filename,
                                entity_filename,
                                type='imports',
                                alias=entity_alias,
                            )
                    else:
                        # Entity might be an attribute inside the module
                        module_path = resolve_module(module_name, repo_path)
                        if module_path:
                            imp_filename = os.path.relpath(module_path, repo_path)
                            node = f'{imp_filename}:{entity_name}'
                            if graph.has_node(node):
                                graph.add_edge(
                                    filename, node, type='imports', alias=entity_alias
                                )
                            elif graph.has_node(imp_filename):
                                graph.add_edge(
                                    filename,
                                    imp_filename,
                                    type='imports',
                                    alias=entity_alias,
                                )

    global_name_dict = defaultdict(list)
    if global_import:
        for node in graph.nodes():
            node_name = node.split(':')[-1].split('.')[-1]
            global_name_dict[node_name].append(node)

    # add 'invokes' edges
    for node, attributes in graph.nodes(data=True):
        if attributes.get('type') not in ['class', 'function']:
            continue

        caller_code_tree = ast.parse(graph.nodes[node]['code'])

        callee_nodes, callee_alias = find_all_possible_callee(node, graph)

        if fuzzy_search:
            callee_name_dict = defaultdict(list)
            for callee_node in set(callee_nodes):
                callee_name = callee_node.split(':')[-1].split('.')[-1]
                callee_name_dict[callee_name].append(callee_node)
            for alias, callee_node in callee_alias.items():
                callee_name_dict[alias].append(callee_node)
        else:
            # Create name -> node dict, for nodes with the same suffix, keep the nearest node
            callee_name_dict = {
                callee_node.split(':')[-1].split('.')[-1]: callee_node
                for callee_node in callee_nodes[::-1]
            }
            callee_name_dict.update(callee_alias)

        if attributes.get('type') == 'class':
            invokes = analyze_init(node, caller_code_tree)
        else:
            invokes = analyze_invokes(node, caller_code_tree)

        # 在图中添加invokes关系的边
        for callee_name in set(invokes):
            callee_node = callee_name_dict.get(callee_name)
            if callee_node:
                if isinstance(callee_node, list):
                    for callee in callee_node:
                        graph.add_edge(node, callee, type='invokes')
                else:
                    graph.add_edge(node, callee_node, type='invokes')
            elif global_import:
                # search from global name dict
                global_fuzzy_nodes = global_name_dict.get(callee_name)
                if global_fuzzy_nodes:
                    for global_fuzzy_node in global_fuzzy_nodes:
                        graph.add_edge(node, global_fuzzy_node, type='invokes')

    return graph


def get_inner_nodes(query_node, src_node, graph):
    inner_nodes = []
    for _, dst_node, attr in graph.edges(src_node, data=True):
        if attr['type'] == 'contains' and dst_node != query_node:
            inner_nodes.append(dst_node)
            if (
                graph.nodes[dst_node]['type'] == 'class'
            ):  # only include class's inner nodes
                inner_nodes.extend(get_inner_nodes(query_node, dst_node, graph))
    return inner_nodes


def find_all_possible_callee(node, graph):
    callee_nodes, callee_alias = [], {}
    cur_node = node
    pre_node = node

    def find_next(_cur_node):
        for predecessor in graph.predecessors(_cur_node):
            for key, attr in graph.get_edge_data(predecessor, _cur_node).items():
                if attr['type'] == 'contains':
                    return predecessor

    while True:
        callee_nodes.extend(get_inner_nodes(pre_node, cur_node, graph))

        if graph.nodes[cur_node]['type'] == 'file':
            # if cur_node == 'astropy/cosmology/connect.py':
            #     breakpoint()

            # check recursive imported files
            file_list: List[Any] = []
            file_stack = [cur_node]
            while len(file_stack) > 0:
                for _, dst_node, attr in graph.edges(file_stack.pop(), data=True):
                    if attr['type'] == 'imports' and dst_node not in file_list + [
                        cur_node
                    ]:
                        if graph.nodes[dst_node][
                            'type'
                        ] == 'file' and dst_node.endswith('__init__.py'):
                            file_list.append(dst_node)
                            file_stack.append(dst_node)
            for file in file_list:
                callee_nodes.extend(get_inner_nodes(cur_node, file, graph))
                for _, dst_node, attr in graph.edges(file, data=True):
                    if attr['type'] == 'imports':
                        if attr['alias'] is not None:
                            callee_alias[attr['alias']] = dst_node
                        if graph.nodes[dst_node]['type'] in ['file', 'class']:
                            callee_nodes.extend(get_inner_nodes(file, dst_node, graph))
                        if graph.nodes[dst_node]['type'] in ['function', 'class']:
                            callee_nodes.append(dst_node)

            # if cur_node == 'astropy/cosmology/connect.py':
            #     breakpoint()

            # check imported functions and classes
            for _, dst_node, attr in graph.edges(cur_node, data=True):
                if attr['type'] == 'imports':
                    if attr['alias'] is not None:
                        callee_alias[attr['alias']] = dst_node
                    if graph.nodes[dst_node]['type'] in ['file', 'class']:
                        callee_nodes.extend(get_inner_nodes(cur_node, dst_node, graph))
                    if graph.nodes[dst_node]['type'] in ['function', 'class']:
                        callee_nodes.append(dst_node)
            break

        pre_node = cur_node
        cur_node = find_next(cur_node)

    return callee_nodes, callee_alias


def analyze_init(node, code_tree):
    caller_name = node.split(':')[-1].split('.')[-1]

    # 存储找到的调用关系
    invokes = []

    def add_invoke(func_name):
        # if func_name in callee_names:
        invokes.append(func_name)

    def process_decorator_node(_decorator_node):
        if isinstance(_decorator_node, ast.Name):
            add_invoke(_decorator_node.id)
        else:
            for _sub_node in ast.walk(_decorator_node):
                if isinstance(_sub_node, ast.Call) and isinstance(
                    _sub_node.func, ast.Name
                ):
                    add_invoke(_sub_node.func.id)
                elif isinstance(_sub_node, ast.Attribute):
                    add_invoke(_sub_node.attr)

    for ast_node in ast.walk(code_tree):
        if isinstance(ast_node, ast.ClassDef) and ast_node.name == caller_name:
            for decorator_node in ast_node.decorator_list:
                process_decorator_node(decorator_node)

            for body_item in ast_node.body:
                if (
                    isinstance(body_item, ast.FunctionDef)
                    and body_item.name == '__init__'
                ):
                    for decorator_node in body_item.decorator_list:
                        process_decorator_node(decorator_node)

                    for sub_node in ast.walk(body_item):
                        if isinstance(sub_node, ast.Call):
                            if isinstance(sub_node.func, ast.Name):  # 普通函数或类
                                add_invoke(sub_node.func.id)
                            if isinstance(sub_node.func, ast.Attribute):  # 成员函数
                                add_invoke(sub_node.func.attr)
                    break
            break

    return invokes


def analyze_invokes(node, code_tree):
    caller_name = node.split(':')[-1].split('.')[-1]

    # 存储找到的调用关系
    invokes = []

    def add_invoke(func_name):
        # if func_name in callee_names:
        invokes.append(func_name)

    def process_decorator_node(_decorator_node):
        if isinstance(_decorator_node, ast.Name):
            add_invoke(_decorator_node.id)
        else:
            for _sub_node in ast.walk(_decorator_node):
                if isinstance(_sub_node, ast.Call) and isinstance(
                    _sub_node.func, ast.Name
                ):
                    add_invoke(_sub_node.func.id)
                elif isinstance(_sub_node, ast.Attribute):
                    add_invoke(_sub_node.attr)

    def traverse_call(_node):
        for child in ast.iter_child_nodes(_node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Skip inner function/class definition
                continue
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    add_invoke(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    add_invoke(child.func.attr)
            # Recursively traverse child nodes
            traverse_call(child)

    # 遍历 AST 节点以找到调用关系
    for ast_node in ast.walk(code_tree):
        if (
            isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and ast_node.name == caller_name
        ):
            # 遍历函数装饰器
            for decorator_node in ast_node.decorator_list:
                process_decorator_node(decorator_node)

            # 遍历函数体内的所有invoke子节点 (不包括内部函数、类)
            traverse_call(ast_node)
            break

    return invokes


def visualize_graph(G):
    node_types = set(nx.get_node_attributes(G, 'type').values())
    node_shapes = {'class': 'o', 'function': 's', 'file': 'D'}
    node_colors = {'class': 'lightgreen', 'function': 'lightblue', 'file': 'lightgrey'}

    edge_types = set(nx.get_edge_attributes(G, 'type').values())
    edge_colors = {
        'imports': 'forestgreen',
        'contains': 'skyblue',
        'invokes': 'magenta',
    }
    edge_styles = {'imports': 'solid', 'contains': 'dashed', 'invokes': 'dotted'}

    # pos = nx.spring_layout(G, k=2, iterations=50)
    pos = nx.shell_layout(G)
    # pos = nx.circular_layout(G)

    plt.figure(figsize=(20, 20))
    plt.margins(0.1)  # Add padding around the plot

    # Draw nodes with different shapes and colors based on their type
    for ntype in node_types:
        nodelist = [n for n, d in G.nodes(data=True) if d['type'] == ntype]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_shape=node_shapes[ntype],
            node_color=node_colors[ntype],
            node_size=700,
            label=ntype,
        )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

    # Group edges between the same pair of nodes
    edge_groups: Dict[Tuple, List] = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        if (u, v) not in edge_groups:
            edge_groups[(u, v)] = []
        edge_groups[(u, v)].append((key, data))

    # Draw edges with adjusted 'rad' values
    for (u, v), edges in edge_groups.items():
        num_edges = len(edges)
        for i, (key, data) in enumerate(edges):
            edge_type = data['type']
            # Adjust 'rad' to spread the edges
            rad = 0.1 * (i - (num_edges - 1) / 2)
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                edge_color=edge_colors[edge_type],
                style=edge_styles[edge_type],
                connectionstyle=f'arc3,rad={rad}',
                arrows=True,
                arrowstyle='-|>',
                arrowsize=15,
                min_source_margin=15,
                min_target_margin=15,
            )

    # Create legends for edge types and node types
    edge_legend_elements = [
        Line2D(
            [0],
            [0],
            color=edge_colors[etype],
            lw=2,
            linestyle=edge_styles[etype],
            label=etype,
        )
        for etype in edge_types
    ]
    node_legend_elements = [
        Line2D(
            [0],
            [0],
            marker=node_shapes[ntype],
            color='w',
            label=ntype,
            markerfacecolor=node_colors[ntype],
            markersize=15,
        )
        for ntype in node_types
    ]

    # Combine legends
    plt.legend(handles=edge_legend_elements + node_legend_elements, loc='upper left')
    plt.axis('off')
    plt.savefig('plot.png')


def main():
    # Generate Dependency Graph
    graph = build_graph_v2(args.repo_path, global_import=args.global_import)

    # G = build_graph(args.repo_path)
    # add_edges(G)
    # G = convert_edges_to_invokes(G)

    if args.visualize:
        visualize_graph(graph)

    import_list = []
    edge_types = []
    for u, v, data in graph.edges(data=True):
        if data['type'] == 'imports':
            import_list.append((u, v))
        edge_types.append(data['type'])
    print(Counter(edge_types))

    # import_list2 = []
    # edge_types2 = []
    # for u, v, data in G.edges(data=True):
    #     if data['type'] == 'imports':
    #         import_list2.append((u, v))
    #     edge_types2.append(data['type'])
    # print(Counter(edge_types2))

    node_types = []
    for node, data in graph.nodes(data=True):
        node_types.append(data['type'])
    print(Counter(node_types))

    # breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--repo_path', type=str, default='DATA/repo/astropy__astropy-12907'
    )
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--global_import', action='store_true')
    args = parser.parse_args()

    main()
