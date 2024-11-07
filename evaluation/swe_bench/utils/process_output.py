def add_new_file(new_file, valid_files, found_files=None):
    if new_file in valid_files and new_file not in found_files:
        found_files.append(new_file)
    return found_files


def get_loc_related_dict_from_raw_output(
    raw_output, valid_files, found_files=None, loc_dict=None
):
    assert valid_files
    # Remove the triple backticks and any surrounding whitespace
    raw_output = raw_output.strip('` \n')

    # Initialize lists
    if not found_files:
        found_files = []
    if not loc_dict:
        loc_dict = {}

    # Split the input data into lines
    lines = raw_output.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            current_file = None
            continue  # Skip empty lines

        if line.endswith('.py'):
            if line not in valid_files:
                current_file = None
                continue
            # It's a file name
            current_file = line
            if current_file not in found_files:
                found_files.append(current_file)
            if current_file not in loc_dict:
                loc_dict[current_file] = []
        elif line.startswith(('function:', 'class:')):
            # It's a function or class definition
            if current_file:
                loc_dict[current_file].append(line)
    return found_files, loc_dict


def get_additional_artifact_loc_related_from_dict(found_files, found_related_locs):
    files = [f for f in found_files if f in found_related_locs]
    output = '```\n'

    for file, locs in zip(files, found_related_locs):
        output += f'{file}\n'
        for loc in locs:
            output += f'{loc}\n'
        output += '\n'
    output += '```'

    additional_artifact_loc_related = [{'raw_output': output}]
    return additional_artifact_loc_related


def get_loc_edit_dict_from_raw_sample_output(
    data, valid_files, file_list=None, loc_related_dict=None, loc_edit_dict=None
):
    valid_top_folder = []
    for fn in valid_files:
        folder = fn.split('/')[0]
        if folder not in valid_top_folder:
            valid_top_folder.append(folder)

    # Remove the triple backticks and any surrounding whitespace
    data = data.strip('` \n')

    # Initialize lists
    if not file_list:
        file_list = []
    if not loc_related_dict:
        loc_related_dict = dict()
    if not loc_edit_dict:
        # Initialize the dictionary to store the edit file information
        loc_edit_dict = dict()

    current_file = None
    current_related = None
    # current_data = None
    # Split the input data into lines
    lines = data.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            # current_file = None
            # current_related = None
            # current_data = []
            continue  # Skip empty lines

        if line.endswith('.py'):
            fn = extract_python_file_path(line, valid_top_folder)
            if not fn or fn not in valid_files:
                current_file = None
                current_related = None
                continue

            current_file = fn
            current_related = None
            if current_file not in file_list:
                file_list.append(current_file)
            if current_file not in loc_related_dict:
                loc_related_dict[current_file] = []
            if current_file not in loc_edit_dict:
                loc_edit_dict[current_file] = {}

        elif line.startswith(('function:', 'class:', 'method:', 'variable:')):
            if current_file:
                current_related = line
                if current_related not in loc_related_dict[current_file]:
                    loc_related_dict[current_file].append(current_related)
                if current_related not in loc_edit_dict[current_file]:
                    loc_edit_dict[current_file][current_related] = []
        elif line.startswith('line'):
            # It's part of the function/class/line data
            if current_file and current_related:
                loc_edit_dict[current_file][current_related].append(line)
            elif current_file:
                if '' not in loc_edit_dict[current_file]:
                    loc_edit_dict[current_file][''] = []
                loc_edit_dict[current_file][''].append(line)
    return file_list, loc_related_dict, loc_edit_dict


def get_loc_edit_dict_from_raw_output(
    raw_output, valid_files, file_list=None, loc_related_dict=None, all_results=None
):
    # all_results = [dict() for i in range(raw_output)]
    found_files = []
    if not all_results:
        all_results = [dict() for i in range(len(raw_output))]
    else:
        assert len(all_results) == len(raw_output)
    all_loc_related_dict = []

    for i, sample in enumerate(raw_output):
        file_list, loc_related_dict, loc_edit_dict = (
            get_loc_edit_dict_from_raw_sample_output(
                sample,
                valid_files,
                # file_list, loc_related_dict,
                loc_edit_dict=all_results[i],
            )
        )
        all_results[i] = loc_edit_dict
        found_files.append(file_list)
        all_loc_related_dict.append(
            loc_related_dict
        )  # TODO: process the loc_related variables
    return found_files, all_loc_related_dict, all_results


def convert_to_loc_related_list(loc_related_dict, file_list):
    loc_related_list = []
    for file in file_list:
        if file in loc_related_dict:
            loc_related_list.append(['\n'.join(loc_related_dict[file])])
        else:
            loc_related_list.append([''])
    return loc_related_list


def convert_to_loc_edit_list(loc_edit_dict, file_list):
    if not isinstance(loc_edit_dict, list):
        loc_edit_dict = [loc_edit_dict]
        file_list = [file_list]

    loc_edit_list = []
    for i, sample in enumerate(loc_edit_dict):
        sample_list = []
        for file in file_list[i]:
            data = []
            if file in sample:
                for modual in sample[file]:
                    data.append(modual)
                    data += sample[file][modual]
                sample_list.append(['\n'.join(data)])
            else:
                sample_list.append([''])
        loc_edit_list.append(sample_list)
    return loc_edit_list


import re


def extract_python_file_path(line, valid_folders):
    """
    Extracts the Python file path from a given line of text.

    Parameters:
    - line (str): A line of text that may contain a Python file path.

    Returns:
    - str or None: The extracted Python file path if found; otherwise, None.
    """
    # Define a regular expression pattern to match file paths ending with .py
    # The pattern looks for sequences of characters that can include letters, numbers,
    # underscores, hyphens, dots, or slashes, ending with '.py'
    pattern = r'[\w\./-]+\.py'

    # Search for the pattern in the line
    match = re.search(pattern, line)

    if match:
        matched_fp = match.group(0)
        start_index = len(matched_fp)
        for folder in valid_folders:
            if folder in matched_fp:
                cur_start_index = matched_fp.index(folder)
                if cur_start_index < start_index:
                    start_index = cur_start_index
        if start_index < len(matched_fp):
            return matched_fp[start_index:]  # Return the max matched file path
        return None
    else:
        return None  # Return None if no match is found


# import re
def extract_result(summary):
    pattern = r'```(.*?)```'
    match = re.search(pattern, summary, re.DOTALL)

    # Extract and format the result if a match is found
    if match:
        result = f'```{match.group(1)}```'
    else:
        result = ''
        print('No match found')
    return result


# output_loc = extract_result(summary_output_loc)


def merge_sample_locations(found_files, found_edit_locs):
    merged_found_files = []
    merged_found_edit_locs = dict()

    for sample_files in found_files:
        for f in sample_files:
            if f not in merged_found_files:
                merged_found_files.append(f)
                merged_found_edit_locs[f] = []

    for i, sample in enumerate(found_edit_locs):
        for j, loc_dict in enumerate(sample):
            merged_found_edit_locs[found_files[i][j]] += loc_dict

    merged_loc_edit_list = []
    for i, file in enumerate(merged_found_files):
        merged_loc_edit_list.append(['\n'.join(merged_found_edit_locs[file])])

    # merged_loc_edit_list = convert_to_loc_edit_list([merged_found_edit_locs],[merged_found_files])
    return merged_found_files, merged_loc_edit_list
