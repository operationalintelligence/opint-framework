"""
This file provide wrappers and abstract implementation of the following spec:

https://docs.google.com/document/d/1qztk6n6_OYubDKynQL05G0H9Yuwf7CX8K-SFJyeE4z0/edit#

To set or use any additional parameters in your functions of the
Vectorization, Tokenization, Clusterization classes, use the self.ctx dict.

"""
from abc import ABC, abstractmethod
import traceback


class PreProcessingError(Exception):
    pass


class NLPAdapter(ABC):
    """ A wrapper class to interface various NLP log analysis algorithms """

    def __init__(self, name, vectorization, tokenization, clusterization):
        super(NLPAdapter, self).__init__()
        self.name = name
        self.vectorization = vectorization
        self.tokenization = tokenization
        self.clusterization = clusterization
        self.context = {}

    @abstractmethod
    def pre_process(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def post_process(self):
        raise NotImplementedError

    @abstractmethod
    def execute(self):
        try:
            print(f"NLP Adapter - {self.name}: Pre Processing input data")
            self.pre_process()
        except Exception as error:
            print(f"NLP Adapter - {self.name}: Pre Processing failed: {str(error)}")
            traceback.print_exc()
        try:
            print(f"NLP Adapter - {self.name}: Executing vectorization, tokenization and clusterization")
            self.run()
        except Exception as error:
            print(f"NLP Adapter - {self.name}: Log analysis failed: {str(error)}")
            traceback.print_exc()
        try:
            print(f"NLP Adapter - {self.name}: Post processing")
        except Exception as error:
            print(f"NLP Adapter - {self.name}: Post Processing failed: {str(error)}")
            traceback.print_exc()


class Vectorization(ABC):

    def __init__(self, ctx):
        super(Vectorization, self).__init__()
        self.ctx = ctx

    def data_preparataion(self, messages):
        """
        Prepare data for tokenization. Optional procedure. It can be used to clean initial log messages from digits, UIDs, and other rare terms.
        For example, line numbers are definitely shouldn’t be utilized in training vector model.

        **Note**: Use the self.ctx to set or fetch additional parameters required by your implementation

        :param messages: log messages. For instance, [Error message at line 100, Error message at line 200, Error message at line 300]
        :type messages: array of strings or log file.
        :return: cleaned messages. For instance, [Error message at line ((*)), Error message at line ((*)), Error message at line ((*))]
        """
        pass

    @abstractmethod
    def train_model(self, messages, path_to_model, embedding_size=300, window=7, min_count=1, workers=1,
                    training_algorithm=1, iter=10):
        """
        Train word2vec model and save it to a file for further usage

        :param list[str]/filepath messages: log messages
        :param str path_to_model: path to file, where word2vec model must be stored
        :param int embedding_size: size of embedding vector
        :param int window: size of slicing window
        :param int min_count: Ignores all words with total frequency lower than this
        :param int workers: number of threads
        :param int training_algorithm: 0 for CBOW, 1 for Skip-Gram
        :param int iter: number of iterations over the corpus
        :return filename for word2vec model i.e trained word2vec model saved in file
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path_to_model):
        """
        Load pre-trained word2vec model from file and save it to python’s object.
        :param str path_to_model: path to file, where word2vec model is stored
        :return word2vec object
        """
        raise NotImplementedError

    @abstractmethod
    def update_model(self, path_to_model, tokenized):
        """
        Load pre-trained word2vec from file, update it, and return word2vec object.
        :param str path_to_model: path to file, where word2vec model is stored
        :param list(list(str)) tokenized: new tokenized log messages
        :return updated word2vec object
        """
        raise NotImplementedError

    @abstractmethod
    def vectorize_messages(self, word2vec, tokenized):
        """
        Calculate the average of words representation for each log message.
        :param word2vec: word2vec model
        :param list(list(str)) tokenized: tokenized log messages
        :return np.array: averaged word vector representation for each log message
        """
        raise NotImplementedError


class Tokenization(ABC):
    def __init__(self, ctx):
        super(Tokenization, self).__init__()
        self.ctx = ctx

    @abstractmethod
    def tokenize_messages(self, tokenizer, messages):
        """
        As an input this stage can take array of strings with log messages or just log file (which then is converted to array of strings).
        One line == one log message

        There are many types of tokenization, implemented in various libraries.
        For example, nltk(python) has TreeBankWordTokenizer, WordPunctTokenizer, StringTokenizer, TweetTokenizer, and some other.
        Pyonmttok - is OpenNMT Tokenizer (https://github.com/OpenNMT/Tokenizer) - is a fast, generic, and customizable text tokenization library for C++ and Python. It provides conservative and aggressive tokenization with different parameters. Pyonmttok allows reversible tokenization, marking joints or spaces by annotating tokens or injecting modifier characters.
        Example Input: [Error message at line 100, Error message at line 200, Error message at line 300]
        Example Output: [[error, message, at, line 100], [error, message, at, line, 200], [error, message, at, line, 300]]

        :param object tokenizer: choose the type of tokenization and available attributes. For instance pyonmttok
        :param list(str) messages: log messages
        :return: tokenized of list(list(str)) which is an array of tokenized messages
        """
        return NotImplementedError

    @abstractmethod
    def detokenize_messages(self, tokenizer, tokenized):
        """
        Example Input (tokenized): [[error, message, at, line ((*))],  [error, message, at, line, ((*))], [error, message, at, line, ((*))]]
        Example Output: [Error message at line ((*)), Error message at line ((*)), Error message at line ((*))]

        :param object tokenizer:  choose the type of tokenization with attributes. For eg, pyonmttok
        :param list(list(str)) tokenized: tokenized log messages
        :return: list(str) patterns: Array of patterns
        """
        return NotImplementedError

    @abstractmethod
    def clean_tokens(self, tokenized):
        """
        Clean tokens from stop words, punctuation, digits (if needed) or some regexp.
        Unnecessary tokens are removed from array.

        Example Input: [[error, message, at, line 100], [error, message, at, line, 200], [error, message, at, line, 300]]
        Example Output: [[error, message, line], [error, message, line], [error, message, line]]

        :param list(list(str)) tokenized: array of tokenized messages
        :return: list(list(str)) cleaned_tokenized: array of cleaned tokenized messages without stop_words and punctuation
        """
        return NotImplementedError

    @abstractmethod
    def get_vocabulary(self, tokenized):
        """
        Vocabulary = all unique tokens of analyzed corpus of log messages
        Example Input: [[error, message, at, line 100], [error, message, at, line, 200], [error, message, at, line, 300]]
        Example Output: [error, message, at, line, 100, 200, 300]
        :param list(list(str)) tokenized: array of tokenized messages
        :return: list vocabulary: list of unique tokens
        """
        return NotImplementedError

    @abstractmethod
    def tokenize_string(self, tokenizer, row):
        """
        Tokenization of single log message (single string)
        Example input (row):  ‘error message at line 100’
        Example output: [error, message, at, line, 100]
        :param object tokenizer: type of tokenizer. like pyonmttok
        :param string row: single log message
        :return: list tokens: tokenized string (array of tokens)
        """
        return NotImplementedError

    @abstractmethod
    def detokenize_string(self, tokenizer, sequence):
        """
        Detokenization of an array of tokens.
        :param object tokenizer: type of tokenizer like pyonmttok
        :param list sequence: tokenized string (log message)
        :return: str pattern: detokenized log message
        """
        return NotImplementedError


class Clustering(ABC):
    def __init__(self, ctx):
        super(Clustering, self).__init__()
        self.ctx = ctx

    @abstractmethod
    def data_preparataion(self, messages):
        """
        Prepare data for clustering. Optional procedure. It can be used to convert input features to proper data formats (e.g. pyspark friendly)

        **Note**: Use the self.ctx to set or fetch additional parameters required by your implementation

        :param messages: dataframe of log messages
        :type messages: array of strings or log file or pyspark.sql.dataframe.DataFrame.
        :return: data to feed into the clustering algorithm
        """
        pass

    @abstractmethod
    def train_model(self, messages, path_to_model, **kwargs):
        """
        Train clustering algorithm and (optionally) save it for further usage

        :param list[str]/filepath messages: log messages
        :param str path_to_model: path to file, where clustering model must be stored
        :param int embedding_size: size of embedding vector
        :param optinal kwargs: additional params specific for the clustering algorithm of the child class
        :return trained model (or reference to it), possibly with some performance metrics attached
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path_to_model):
        """
        Load pre-trained clustering model and return model object.
        :param str path_to_model: path to file, where clustering model is stored
        :return model object
        """
        raise NotImplementedError

    @abstractmethod
    def update_model(self, path_to_model, tokenized):
        """
        Load pre-trained clustering model, update it, and return model object.
        :param str path_to_model: path to file, where word2vec model is stored
        :param list(list(str)) tokenized: new tokenized log messages
        :return updated word2vec object
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, tokenized, **kwargs):
        """
        Apply the pre-trained algorithm in model and output the clusters discovered in the tokenized data.
        :param model: clustering model
        :param tokenized: tokenized log messages
        :return data with cluster label prediction according to the clustering model
        """
        raise NotImplementedError
