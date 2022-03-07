import csv
import random
import os
from pathlib import Path
from collections import Counter

from nltk import tokenize as nltk_tokenize

from data_management import common
from data_management.common import MissingMetadataError
from data_management.common import load_csv_to_dict
from data_management.document import Document


class Corpus:

    """The corpus class is used to load the metadata and full
    texts of all documents in a corpus

    Once loaded, each corpus contains a list of Document objects

    :param path_to_files: Must be either the path to a directory of txt files or an already-pickled corpus
    :param name: Optional name of the corpus, for ease of use and readability
    :param csv_path: Optional path to a csv metadata file
    :param pickle_on_load: Filepath to save a pickled copy of the corpus
    """

    def __init__(self, path_to_files, name=None, drop_missing=False):
        self.name = name
        
        if path_to_files == None: # return empty corpus
            self.documents = []
            self.metadata_fields = set()
            self.generate_properties()
            return

        self.documents, self.metadata_fields = self._load_documents_and_metadata(path_to_files, drop_missing)
        self.generate_properties()

    def generate_properties(self):
        self.filenames = set( document.filename for document in self.documents )

        self.documents = sorted(self.documents)

        # sanity check to ensure every metadata entry has a unique file
        assert( len(self.filenames) == len(self.documents) )

    def _load_documents_and_metadata(self, path_to_files, drop_missing):
        """
        Loads documents into the corpus with metadata from a csv file at a given location
        """

        documents = []
        metadata_fields = set()


        metadata_path = Path(path_to_files, '_metadata.csv')
        loaded_document_filenames = []

        if os.path.exists( metadata_path ): # there is metadata to read

            # load documents from metadata into list
            for doc_metadata in load_csv_to_dict(metadata_path):


                
                
                filename = doc_metadata['filename']
                if not filename.endswith('.txt'):
                    filename += '.txt'

                doc_metadata['name'] = self.name
                
                try:
                    this_document = Document(Path( path_to_files, filename ), doc_metadata)
                    documents.append(this_document)
                    loaded_document_filenames.append(this_document.filename)
                    metadata_fields.update(list(doc_metadata))
                except FileNotFoundError as err:
                    if drop_missing:
                        print(f"Dropping missing file: {doc_metadata['filepath']}")
                        continue
                    else:
                        raise err

            # determine if files in directory were skipped
            all_txt_files = [f for f in os.listdir(path_to_files) if f.endswith('.txt')]
            #print(all_txt_files)
            num_loaded = len(documents)
            #print(num_loaded)
            num_txt_files = len(all_txt_files)
            #print(num_txt_files)
            if num_loaded != num_txt_files:
                # some txt files aren't in the metadata, so issue a warning
                # we don't need to handle the inverse case, because that
                # will have broken the document init above

                print(
                    f'WARNING: The following .txt files were not loaded because they '
                    + 'are not your metadata csv:\n'
                    + str(list(set(all_txt_files) - set(loaded_document_filenames)))
                    + '\nYou may want to check that your metadata matches your files '
                    + 'to avoid incorrect results.'
                )

            # metadata.remove('filepath') #TODO: find a better way to do this
            return documents, metadata_fields

        else: # no metadata
            files = os.listdir(path_to_files)
            metadata_fields.add('filename')
            ignored = []
            documents = []
            for filename in files:
                if filename.endswith('.txt'):
                    metadata_dict = {'filename': filename}
                    documents.append( Document( Path(path_to_files, filename), metadata_dict) )
                else:
                    ignored.append(filename)

            if len(documents) == 0:  # path led to directory with no .txt files
                raise ValueError(f'path_to_files must lead to a previously pickled corpus or directory of .txt files')
            elif ignored:
                print(
                    'WARNING: the following files were not loaded because they are not .txt files.\n'
                   + str(ignored) + '\n'
                   + 'If you would like to analyze the text in these files, convert these files to '
                   + '.txt and create a new Corpus.'
                )

            return documents, metadata_fields

    def get_wordcount_counter(self):
        """
        This function returns a Counter object that stores how many times each word appears in the corpus.

        :return: Python Counter object
        """
        corpus_counter = Counter()
        for current_document in self.documents:
            document_counter = current_document.get_wordcount_counter()
            corpus_counter += document_counter
        return corpus_counter

    def get_field_vals(self, field):
        """
        This function returns a sorted list of the values present in the corpus for a given metadata field.

        :param field: field to search for (i.e. 'location', 'author_gender', etc.)
        :return: list of strings
        """

        if field not in self.metadata_fields:
            raise MissingMetadataError([field])

        values = set()
        for document in self.documents:
            values.add(getattr(document, field))

        return sorted(list(values))

    def subcorpus(self, metadata_field, field_value):
        """
        Returns a new Corpus object that contains only documents with a given field_value for metadata_field

        :param metadata_field: metadata field to search
        :param field_value: search term
        :return: Corpus object
        """
        

        corpus = self.clone()
        corpus.documents = []

        # adds documents to corpus_copy
        for this_document in self.documents:
            try:
                this_value = getattr(this_document, metadata_field, None)
                if this_value is not None and this_value == field_value:
                    corpus.documents.append(this_document)
            except AttributeError:
                continue

        corpus.generate_properties()
        return corpus

    def multi_filter(self, characteristic_dict):
        """
        Returns a copy of the corpus, but with only the documents that fulfill the metadata parameters passed in by
        characteristic_dict. Multiple metadata keys can be searched at one time, provided that the metadata is
        available for the documents in the corpus.


        :param characteristic_dict: Dictionary with metadata fields as keys and search terms as values
        :return: Corpus object
        """

        corpus_copy = self.clone()
        corpus_copy.documents = []

        for metadata_field in characteristic_dict:
            if metadata_field not in self.metadata_fields:
                raise MissingMetadataError([metadata_field])

        for this_document in self.documents:
            add_document = True
            for metadata_field in characteristic_dict:
                if metadata_field == 'date':
                    if this_document.date != int(characteristic_dict['date']):
                        add_document = False
                else:
                    if getattr(this_document, metadata_field) != characteristic_dict[metadata_field]:
                        add_document = False
            if add_document:
                corpus_copy.documents.append(this_document)

        if not corpus_copy:
            # displays for possible errors in field.value
            err = f'This corpus is empty. You may have mistyped something.'
            raise AttributeError(err)

        return corpus_copy

    def get_document(self, metadata_field, field_val):
        """
        Returns a specific Document object from self.documents that has metadata matching field_val for
        metadata_field.

        This function will only return the first document in self.documents. It should only be used if you're certain
        there is only one match in the Corpus or if you're not picky about which Document you get.  If you want more
        selectivity use **get_document_multiple_fields**, or if you want multiple documents,
        use **subcorpus**.

        :param metadata_field: metadata field to search
        :param field_val: search term
        :return: Document Object
        """

        if metadata_field not in self.metadata_fields:
            raise MissingMetadataError([metadata_field])

        #if metadata_field == "date":
            #field_val = int(field_val)

        for document in self.documents:
            if getattr(document, metadata_field) == field_val:
                return document

        raise ValueError("Document not found")

    def get_sample_text_passages(self, expression, no_passages):
        """
        Returns a specified number of example passages that include a certain expression.

        The number of passages that you request is a maximum number, and this function may return
        fewer if there are limited cases of a passage in the corpus.

        :param expression: expression to search for
        :param no_passages: number of passages to return
        :return: List of passages as strings
        """
        count = 0
        output = []
        phrase = nltk_tokenize.word_tokenize(expression)
        random.seed(expression)
        random_documents = self.documents.copy()
        random.shuffle(random_documents)

        for document in random_documents:
            if count >= no_passages:
                break
            current_document = document.get_tokenized_text()
            for index in range(len(current_document)):
                if current_document[index] == phrase[0]:
                    if current_document[index:index+len(phrase)] == phrase:
                        passage = " ".join(current_document[index-20:index+len(phrase)+20])
                        output.append((document.filename, passage))
                        count += 1

        if len(output) <= no_passages:
            return output
        return output[:no_passages]

    def get_document_multiple_fields(self, metadata_dict):
        """
        Returns a specific Document object from the corpus that has metadata matching a given metadata dict.

        This method will only return the first document in the corpus.  It should only be used if you're certain
        there is only one match in the Corpus or if you're not picky about which Document you get.  If you want
        multiple documents, use **subcorpus**.

        :param metadata_dict: Dictionary with metadata fields as keys and search terms as values
        :return: Document object
        """

        for field in metadata_dict.keys():
            if field not in self.metadata_fields:
                raise MissingMetadataError([field])

        for document in self.documents:
            match = True
            for field, val in metadata_dict.items():
                if getattr(document, field, None) != val:
                    match = False
            if match:
                return document

        raise ValueError("Document not found")

    def update_metadata(self, new_metadata_path):
        """
        Takes a filepath to a csv with new metadata and updates the metadata in the corpus'
        documents accordingly. The new file does not need to contain every metadata field in
        the documents - only the fields that you wish to update.

        NOTE: The csv file must include at least a filename for the documents that will be altered.

        :param new_metadata_path: Path to new metadata csv file
        :return: None
        """
        metadata = set()
        metadata.update(self.metadata_fields)

        if isinstance(new_metadata_path, str):
            new_metadata_path = Path(new_metadata_path)
        if not isinstance(new_metadata_path, Path):
            raise ValueError(f'new_metadata_path must be str or Path object, not type {type(new_metadata_path)}')

        try:
            csv_list = load_csv_to_list(new_metadata_path)
        except FileNotFoundError:
            err = "Could not find the metadata csv file for the "
            err += f"corpus in the expected location ({self.csv_path})."
            raise FileNotFoundError(err)
        csv_reader = csv.DictReader(csv_list)

        for document_metadata in csv_reader:
            document_metadata = dict(document_metadata)
            metadata.update(list(document_metadata))
            try:
                document = self.get_document('filename', document_metadata['filename'])
            except ValueError:
                raise ValueError(f"Document {document_metadata['filename']} not found in corpus")

            document.update_metadata(document_metadata)

        self.metadata_fields = list(metadata)

    def get_metadata(self):
        metadata = []
        for document in self:
            piece = {}
            for field in sorted(list(self.metadata_fields)):
                piece[field] = getattr(document, field, '')
            metadata.append(piece)
        return metadata

    def generate_unique_filename(self, basename):
        if basename == '': basename = 'no author'
        basename = basename.replace('.txt','')
        name = basename + '.txt'
        i = 2
        while name in self.filenames:
            name = f'{basename}-{i}.txt'
            i += 1
        return name

    def save_to(self, path):
        self.generate_properties()

        if not os.path.exists(path):
            os.makedirs(path)

        for document in self:
            # write new file
            doc_path = Path(path, document.filename)
            if os.path.exists(doc_path):
                print(f'WARNING: overwritting file at {doc_path}')
            with open(doc_path , 'w+', newline='', encoding=document.encoding) as file:
                file.write( document.text )
        # save metadata
        common.write(Path(path, '_metadata.csv'), self.get_metadata() )

    def merge(self, other, fuzzy=None, keep_other=False):
        corpus = self.clone()
        other = other.clone()

        removed = 0
        for other_doc in other:
            duplicate = False
            for document in corpus:
                if common.compare(document.text, other_doc.text, fuzzy=fuzzy):
                    print(f'found duplicate in corpus: \n  >{document.filepath} \n  >{other_doc.filepath}')
                    removed += 1
                    
                    #overwrite target_doc with source_doc
                    if keep_other:
                        source, source_doc, target_doc = other, other_doc, document
                    else:
                        source, source_doc, target_doc = corpus, document, other_doc
                    
                    source -= source_doc
                    target_doc.notes = f'Overwritten from {source_doc.filepath}\n\n' + target_doc.text
                    target_doc.text, target_doc.encoding = source_doc.text, source_doc.encoding

                    duplicate = True
                    break
            
            if not duplicate:
                corpus += other_doc
        corpus.metadata_fields.update( other.metadata_fields )
        print( f'removed {removed} duplicates during merge')
        
        corpus.generate_properties()
        return corpus #+ other #generic addition should handle duplicate filenames

    def __len__(self):
        """
        For convenience: returns the number of documents in
        the corpus.

        :return: number of documents in the corpus as an int
        """
        return len(self.documents)

    def __iter__(self):
        """
        Yield each of the documents from the .documents list.

        For convenience.
        """
        for this_document in self.documents:
            yield this_document

    def __eq__(self, other):
        """
        Returns true if both corpora contain the same documents
        Note: ignores differences in the corpus name as that attribute is not used apart from
        initializing a corpus.
        Presumes the documents to be sorted. (They get sorted by the initializer)
        
        :return: bool
        """
        if not isinstance(other, Corpus):
            return False

        if len(self) != len(other):
            return False

        for i in range(len(self)):
            if self.documents[i] != other.documents[i]:
                return False

        return True

    def __add__(self, other):
        """
        Adds two corpora together and returns a copy of the result
        Note: retains the name of the first corpus

        :return: Corpus
        """

        corpus = self.clone()

        if isinstance(other, Corpus):
            for document in other:
                document.filename = corpus.generate_unique_filename( document.filename )
                
                corpus.filenames.add( document.filename )
                corpus.documents.append(document)
            
            corpus.metadata_fields.update( other.metadata_fields )
            corpus.generate_properties()
            return corpus
        elif isinstance(other, Document):
            other = other.clone()
            other.filename = corpus.generate_unique_filename( other.filename )
            
            corpus.filenames.add( other.filename )
            corpus.documents.append(other)
            
            corpus.generate_properties()
            return corpus
        else:
            raise NotImplementedError("Only a Corpus or Document can be added to another Corpus.")


    def __sub__(self, other):
        
        out = self.clone()

        if isinstance(other, Document):
            out.documents.remove(other)
        elif isinstance(other, Corpus):
            for document in other:
                out = out - document

        out.generate_properties()
        return out

    def clone(self):
        """
        Return a copy of the Corpus object
        :return: Corpus object
        """
        from copy import deepcopy

        corpus = deepcopy(self)
        corpus.generate_properties()
        return corpus

