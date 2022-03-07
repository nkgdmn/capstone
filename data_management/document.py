import re
import string
from collections import Counter
from pathlib import Path

from more_itertools import windowed
import nltk
from data_management import common

VITAL_KEYS = ['author', 'performance_work', 'filename', 'language', 'citation', 'date', 'status']

class Document:
    """
    The Document class loads and holds the full text and
    metadata (author, title, publication date, etc.) of a document

    :param metadata_dict: Dictionary with metadata fields as keys and data as values
    """

    def __init__(self, filepath, metadata_dict):
        assert( isinstance(metadata_dict, dict) ) # metadata must be passed in as a dictionary value

        
        self.members = list(metadata_dict.keys())

        for key in metadata_dict:
            if hasattr(self, str(key)):
                raise KeyError(
                    'Key name ', str(key), ' is reserved in the Document class. Please use another name'
                )
            setattr(self, str(key), metadata_dict[key])

        self.filename = common.normalize_string( self.filename )
        if not self.filename.endswith('.txt'):
            self.filename += '.txt'


        # Check if document metadata is complete
        self.complete = True
        for key in VITAL_KEYS:
            if (not hasattr(self, key)) or getattr(self, key) == '': self.complete = False

        self._word_counts_counter = None
        self._word_count = None
        
        self.filepath = filepath
        self.encoding = common.get_text_file_encoding(filepath)
        self.text = self._load_document_text(filepath)

    @property
    def word_count(self):
        """
        Lazy-loading for **Document.word_count** attribute. Returns the number of words in the document.
        The word_count attribute is useful for the get_word_freq function.
        However, it is performance-wise costly, so it's only loaded when it's actually required.

        :return: Number of words in the document's text as an int
        """

        if self._word_count is None:
            self._word_count = len(self.get_tokenized_text())
        return self._word_count

    def __str__(self):
        """
        Overrides python print method for user-defined objects for Document class
        Returns the filename without the extension - author and title word
        :return: str
        """
        name = self.filename[0:len(self.filename) - 4]
        return name

    def __repr__(self):
        '''
        Overrides the built-in __repr__ method
        Returns the object type (Document) and then the filename without the extension
            in <>.

        :return: string
        '''

        return f'<Document ({self.__str__()})>'

    def __eq__(self, other):
        """
        Overload the equality operator to enable comparing and sorting documents. Returns True if the document filenames
        and text are the same.

        :return: bool
        """
        if not isinstance(other, Document):
            raise NotImplementedError("Only a Document can be compared to another Document.")

        attributes_required_to_be_equal = ['filepath']

        for attribute in attributes_required_to_be_equal:
            if not hasattr(other, attribute):
                raise common.MissingMetadataError([attribute], f'{str(other)} lacks attribute {attribute}.')
            if getattr(self, attribute) != getattr(other, attribute):
                return False

        if self.text != other.text:
            return False

        return True

    def __lt__(self, other):
        """
        Overload less than operator to enable comparing and sorting documents.

        Sorts by filenames.

        :return: bool
        """
        if not isinstance(other, Document):
            raise NotImplementedError("Only a Document can be compared to another Document.")

        return self.filename < other.filename

    def __hash__(self):
        """
        Makes the Document object hashable

        :return: hash of the repr
        """

        return hash(repr(self))

    def _clean_quotes(text):
        """
        Scans through the text and replaces all of the smart quotes and apostrophes with their
        "normal" ASCII variants
        >>> from gender_analysis import Document
        >>> smart_text = 'This is a “smart” phrase'
        >>> Document._clean_quotes(smart_text)
        'This is a "smart" phrase'
        :param text: The string to reformat
        :return: A string that is idential to `text`, except with its smart quotes exchanged
        """

        # Define the quotes that will be swapped out
        smart_quotes = {
            '“': '"',
            '”': '"',
            "‘": "'",
            "’": "'",
        }

        # Replace all entries one by one
        output_text = text
        for quote in smart_quotes:
            output_text = output_text.replace(quote, smart_quotes[quote])

        return output_text

    def _load_document_text(self, filepath):
        """
        Loads the text of the document at the filepath specified in initialization.

        :return: str
        """
        try:
            text = common.load_txt_to_string(filepath)
        except FileNotFoundError as original_err:
            err = (
                f'The filename {self.filepath} present in your metadata csv does not exist in your '
               + 'files directory.\nPlease check that your metadata matches your dataset.'
            )
            raise FileNotFoundError(err) from original_err

        return text

    def get_tokenized_text(self):
        """
        Tokenizes the text and returns it as a list of tokens, while removing all punctuation.

        Note: This does not currently properly handle dashes or contractions.

        :return: List of each word in the Document
        """

        # Excluded characters: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~
        
        # excluded_characters = set(string.punctuation)
        # cleaned_text = ''
        # for character in self.text:
        #     if character not in excluded_characters:
        #         cleaned_text += character

        # tokenized_text = cleaned_text.lower().split()
        # return tokenized_text
        
        lowered = self.text.lower()
        clean_text = Document._clean_quotes(lowered)

        parsed = nltk.word_tokenize(clean_text)
        clean_list = [word for word in parsed if word[0].isalpha()]
        
        return clean_list

    def find_quoted_text(self):
        """
        Finds all of the quoted statements in the document text.

        :return: List of strings enclosed in double-quotations
        """
        text_list = self.text.split()
        quotes = []
        current_quote = []
        quote_in_progress = False
        quote_is_paused = False

        for word in text_list:
            if word[0] == "\"":
                quote_in_progress = True
                quote_is_paused = False
                current_quote.append(word)
            elif quote_in_progress:
                if not quote_is_paused:
                    current_quote.append(word)
                if word[-1] == "\"":
                    if word[-2] != ',':
                        quote_in_progress = False
                        quote_is_paused = False
                        quotes.append(' '.join(current_quote))
                        current_quote = []
                    else:
                        quote_is_paused = True

        return quotes

    def get_count_of_word(self, word):
        """
        Returns the number of instances of a word in the text. Not case-sensitive.

        If this is your first time running this method, it may take a moment to perform a count in the document.

        :param word: word to be counted in text
        :return: Number of occurences of the word, as an int
        """

        # If word_counts were not previously initialized, do it now and store it for the future.
        if not self._word_counts_counter:
            self._word_counts_counter = Counter(self.get_tokenized_text())

        return self._word_counts_counter[word]

    def get_wordcount_counter(self):
        """
        Returns a counter object of all of the words in the text.

        If this is your first time running this method, it may take a moment to perform a count in the document.

        :return: Python Counter object
        """

        # If word_counts were not previously initialized, do it now and store it for the future.
        if not self._word_counts_counter:
            self._word_counts_counter = Counter(self.get_tokenized_text())
        return self._word_counts_counter

    def words_associated(self, word):
        """
        Returns a Counter of the words found after a given word.

        In the case of double/repeated words, the counter would include the word itself and the next
        new word.

        Note: words always return lowercase.

        :param word: Single word to search for in the document's text
        :return: a Python Counter() object with {associated_word: occurrences}
        """
        word = word.lower()
        word_count = Counter()
        check = False
        text = self.get_tokenized_text()

        for w in text:
            if check:
                word_count[w] += 1
                check = False
            if w == word:
                check = True
        return word_count

    def get_word_windows(self, search_terms, window_size=2):
        """
        Finds all instances of `word` and returns a counter of the words around it.
        window_size is the number of words before and after to return, so the total window is
        2*window_size + 1.

        This is not case sensitive.

        :param search_terms: String or list of strings to search for
        :param window_size: integer representing number of words to search for in either direction
        :return: Python Counter object
        """

        if isinstance(search_terms, str):
            search_terms = [search_terms]

        search_terms = set(i.lower() for i in search_terms)

        counter = Counter()

        for text_window in windowed(self.get_tokenized_text(), 2 * window_size + 1):
            if text_window[window_size] in search_terms:
                for surrounding_word in text_window:
                    if surrounding_word not in search_terms:
                        counter[surrounding_word] += 1

        return counter

    def get_word_freq(self, word):
        """
        Returns the frequency of appearance of a word in the document

        :param word: str to search for in document
        :return: float representing the portion of words in the text that are the parameter word
        """

        word_frequency = self.get_count_of_word(word) / self.word_count
        return word_frequency

    def get_part_of_speech_tags(self):
        """
        Returns the part of speech tags as a list of tuples. The first part of each tuple is the
        term, the second one the part of speech tag.

        Note: the same word can have a different part of speech tags. In the example below,
        see "refuse" and "permit".

        :return: List of tuples (term, speech_tag)
        """

        text = nltk.word_tokenize(self.text)
        pos_tags = nltk.pos_tag(text)
        return pos_tags

    def update_metadata(self, new_metadata):
        """
        Updates the metadata of the document without requiring a complete reloading of the text and other properties.
        'filename' cannot be updated with this method.

        :param new_metadata: dict of new metadata to apply to the document
        :return: None

        This can be used to correct mistakes in the metadata:
        """

        if not isinstance(new_metadata, dict):
            raise ValueError(f'new_metadata must be a dictionary of metadata keys, not type {type(new_metadata)}')
        if 'filename' in new_metadata and new_metadata['filename'] != self.filename:
            raise KeyError(f'You cannot update the filename of a document; consider removing {str(self)} from the '
                           f'Corpus object and adding the document again with the updated filename')

        for key in new_metadata:
            if key == 'date':
                try:
                    new_metadata[key] = int(new_metadata[key])
                except ValueError:
                    raise ValueError(f"the metadata field 'date' must be a number for document {self.filename}, not "
                                     f"'{new_metadata['date']}'")
            setattr(self, key, new_metadata[key])
    

    def clone(self):
        """
        Return a copy of the Document object
        :return: Document object
        """
        from copy import deepcopy

        return deepcopy(self)
