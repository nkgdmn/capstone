"""A script to read and write csv files"""

import csv
import codecs
import os
from pathlib import Path
import copy
import Levenshtein as lev


def compare(str1, str2, fuzzy=None):
        str1 = normalize_string(str1)
        str2 = normalize_string(str2)
        if str1 == str2: return True
        if str1 in str2: return True
        if str2 in str1: return True

        if fuzzy and lev.ratio(str1, str2) > fuzzy: return True

        return False

def normalize_string(string):
    return string.replace('?', '').strip().lower()

def separate_csv_data(entries, path):
    entries = copy.deepcopy(entries)
    
    
    # Find every unique work
    # works = set( normalize_string(row['performance_work']) for row in entries )

    # make the output directory if it doesnt exist
    if not os.path.exists(path):
        os.makedirs(path)

    # for work in works:
    for rowset in ((complete, 'complete'), (incomplete,'incomplete')):
        # new_path = Path(path, work)
        new_path = Path(path, rowset[1])

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        pieces = []
        file_names = set()
        # for row in entries:
        for row in rowset[0]:
        
            # if normalize_string(row['performance_work']) == work:

                # make a unique filename
            start_name = normalize_string(row['author'])
            i = 1
            if start_name == '': start_name = 'no author'
            file_name = start_name

            while file_name in file_names:
                i += 1
                file_name = f'{start_name}{i}'
                
            file_names.add(file_name)


            with open( Path(new_path, str(file_name) + '.txt'), 'w+', newline='', encoding='utf-8' ) as file:
                file.write( row['filename'] )
            
            row['filename'] = file_name
            pieces.append(row)

        if len(pieces) != 0: write(Path(new_path, '_metadata.csv'), pieces )
        else: print(f'Uh oh mistakes! This work has no pieces attached: {rowset[1]}')




def load_csv_to_dict(path, encoding=None):
    if encoding == None: encoding = get_text_file_encoding(path)
    with open(path, newline='', encoding=encoding, errors="ignore") as csvfile:
        
        out = []

        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                #if statement can be placed here to select certain columns only
                row[key]=value.lower()
            out.append(row)
        #print(out)
        return out
    raise Exception('CSV parsing error. Check the encoding')


def write(path, arr, encoding='utf-8'):
    with open(path, 'w', newline='', encoding=encoding) as csvfile:
        writer = csv.DictWriter(csvfile, arr[0].keys())
        writer.writeheader()
        writer.writerows(arr)


def load_csv_to_list(file_path):
    """
    Loads a csv file from the given filepath and returns its contents as a list of strings.

    :param file_path: str or Path object
    :return: a list of strings

    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> corpus_metadata_path = Path(common.TEST_DATA_PATH, 'sample_novels', 'sample_novels.csv')
    >>> corpus_metadata = load_csv_to_list(corpus_metadata_path)
    >>> type(corpus_metadata)
    <class 'list'>

    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    file_type = file_path.suffix

    if file_type != '.csv':
        raise Exception(
            'Cannot load if current file type is not .csv'
        )
    else:
        file = open(file_path, encoding='utf-8')
        result = file.readlines()

    file.close()
    return result


def load_txt_to_string(file_path):
    """
    Loads a txt file and returns a str representation of it.

    :param file_path: str or Path object
    :return: The file's text as a string

    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> novel_path = Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'austen_persuasion.txt')
    >>> novel_text = load_txt_to_string(novel_path)
    >>> type(novel_text), len(novel_text)
    (<class 'str'>, 486253)

    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    file_type = file_path.suffix

    if file_type != '.txt':
        raise Exception(
            'Cannot load if current file type is not .txt'
        )
    else:
        try:
            encoding = get_text_file_encoding(file_path)
            file = open(file_path, encoding=encoding)
            result = file.read()
        except UnicodeDecodeError as err:
            print(f'Unicode file loading error {file_path}.')
            raise err

    file.close()
    return result


def get_text_file_encoding(filepath):
    """
    Returns the text encoding as a string for a txt file at the given filepath.

    :param filepath: str or Path object
    :return: Name of encoding scheme as a string

    >>> from gender_analysis import common
    >>> from pathlib import Path
    >>> import os
    >>> path=Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'hawthorne_scarlet.txt')
    >>> common.get_text_file_encoding(path)
    'UTF-8-SIG'

    Note: For files containing only ascii characters, this function will return 'ascii' even if
    the file was encoded with utf-8

    >>> import os
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> text = 'here is an ascii text'
    >>> file_path = Path(common.BASE_PATH, 'example_file.txt')
    >>> with codecs.open(file_path, 'w', 'utf-8') as source:
    ...     source.write(text)
    ...     source.close()
    >>> common.get_text_file_encoding(file_path)
    'ascii'
    >>> file_path = Path(common.BASE_PATH, 'example_file.txt')
    >>> os.remove(file_path)

    """
    from chardet.universaldetector import UniversalDetector
    detector = UniversalDetector()

    with open(filepath, 'rb') as file:
        for line in file:
            detector.feed(line)
            # if detector.done:
            #     break
        detector.close()
    return detector.result['encoding']


def convert_text_file_to_new_encoding(source_path, target_path, target_encoding):
    """
    Converts a text file in source_path to the specified encoding in target_encoding

    Note: Currently only supports encodings utf-8, ascii and iso-8859-1

    :param source_path: str or Path
    :param target_path: str or Path
    :param target_encoding: str
    :return: None

    >>> from gender_analysis.common import BASE_PATH
    >>> text = ' ¶¶¶¶ here is a test file'
    >>> source_path = Path(BASE_PATH, 'source_file.txt')
    >>> target_path = Path(BASE_PATH, 'target_file.txt')
    >>> with codecs.open(source_path, 'w', 'iso-8859-1') as source:
    ...     source.write(text)
    >>> get_text_file_encoding(source_path)
    'ISO-8859-1'
    >>> convert_text_file_to_new_encoding(source_path, target_path, target_encoding='utf-8')
    >>> get_text_file_encoding(target_path)
    'utf-8'
    >>> import os
    >>> os.remove(source_path)
    >>> os.remove(target_path)
    """

    valid_encodings = ['utf-8', 'utf8', 'UTF-8-SIG', 'ascii', 'iso-8859-1', 'ISO-8859-1',
                       'Windows-1252']

    # if the source_path or target_path is a string, turn to Path object.
    if isinstance(source_path, str):
        source_path = Path(source_path)
    if isinstance(target_path, str):
        target_path = Path(target_path)

    # check if source and target encodings are valid
    source_encoding = get_text_file_encoding(source_path)
    if source_encoding not in valid_encodings:
        raise ValueError('convert_text_file_to_new_encoding() only supports the following source '
                         f'encodings: {valid_encodings} but not {source_encoding}.')
    if target_encoding not in valid_encodings:
        raise ValueError('convert_text_file_to_new_encoding() only supports the following target '
                         f'encodings: {valid_encodings} but not {target_encoding}.')

    # print warning if filenames don't end in .txt
    if not source_path.parts[-1].endswith('.txt') or not target_path.parts[-1].endswith('.txt'):
        print(f"WARNING: Changing encoding to {target_encoding} on a file that does not end with "
              f".txt. Source: {source_path}. Target: {target_path}")

    with codecs.open(source_path, 'rU', encoding=source_encoding) as source_file:
        text = source_file.read()
    with codecs.open(target_path, 'w', encoding=target_encoding) as target_file:
        target_file.write(text)


class MissingMetadataError(Exception):
    """
    Raised when a function that assumes certain metadata is called on a corpus without that
    metadata
    """
    def __init__(self, metadata_fields, message=None):
        self.metadata_fields = metadata_fields
        self.message = message if message else ''

    def __str__(self):
        metadata_string = ''

        for i, field in enumerate(self.metadata_fields):
            metadata_string += field
            if i != len(self.metadata_fields) - 1:
                metadata_string += ', '

        is_plural = len(self.metadata_fields) > 1

        return (
            'This Corpus is missing the following metadata field' + ('s' if is_plural else '') + ':\n'
            + '    ' + metadata_string + '\n'
            + self.message + ('\n' if self.message else '')
            + 'In order to run this function, you must create a new metadata csv\n'
            + 'with ' + ('these ' if is_plural else 'this ') + 'field' + ('s ' if is_plural else ' ')
            + 'and run Corpus.update_metadata().'
        )
