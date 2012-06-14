import bz2
import pickle

import numpy as np

__all__ = ['BaseCorpus', 'Corpus']


class BaseCorpus(object):
    """
    A BaseCorpus object stores a corpus along with its tokenizations
    (e.g., as sentences, paragraphs or documents).

    BaseCorpus aims to provide an efficient method for viewing the
    corpus in tokenized form (i.e., without copying). It currently
    achieves this by storing the corpus as a numpy `ndarray`. Viewing
    a tokenization is carried out using the `view` facilities of numpy
    ndarrays. See documentation on `numpy.ndarray` for further
    details.
    
    Parameters
    ----------
    corpus : array-like
        Array, typically of strings or integers, of atomic terms (or
        tokens) making up the corpus
    tokens : dict-like with 1-D integer array-like values, optional
        Each key in `tokens` is a name of a tokenization. Each value
        in `tokens` is an array containing the indices marking the
        token boundaries. A value in `tokens` is intended for use as a
        value for the `indices_or_sections` parameter in
        `numpy.split`. Default is `None`.
    dtype : data-type, optional
        The data-type used to interpret the corpus. If omitted, the
        data-type is determined by `numpy.asarray`. Default is `None`.

    Attributes
    ----------
    corpus : 1-D array
        Stores the value of the `corpus` parameter after it has been
        cast to a array of data-type `dtype` (if provided).
    terms : 1-D array
        The indexed set of atomic terms appearing in `corpus`.
        Computed on initialization by `_extract_terms`.
    tokens : dict with 1-D numpy integer arrays as values 
        Stores the value of the `tokens` parameter. On initialization,
        tokenizations are validated by `validate_tokens` and cast as
        numpy integer arrays

    Methods
    -------
    view_tokens
        Takes a name of tokenization and returns a view of the corpus
        tokenized accordingly.
    validate_tokens
        Static method. Takes a BaseCorpus-like object and verifies
        that the tokenizations are sorted and in range.
    extract_terms
        Static method. Takes an array-like object and returns an
        indexed set of the elements in the object as a 1-D numpy
        array.


    Examples
    --------

    >>> corpus = ['the', 'dog', 'chased', 'the', 'cat',
                  'the', 'cat', 'ran', 'away']
    >>> tokens = {'sentences': [5]}

    >>> from inphosemantics.corpus import BaseCorpus
    >>> c = BaseCorpus(corpus, tokens=tokens)
    >>> c.corpus
    array(['the', 'dog', 'chased', 'the', 'cat', 'the', 'cat',
           'ran', 'away'], dtype='|S6')
           
    >>> c.terms
    array(['ran', 'away', 'chased', 'dog', 'cat', 'the'],
          dtype='|S6')

    >>> c.view_tokens('sentences')
    [array(['the', 'dog', 'chased', 'the', 'cat'],
          dtype='|S6'),
     array(['the', 'cat', 'ran', 'away'],
          dtype='|S6')]

    """
    def __init__(self, corpus, tokens=None, dtype=None):

        self.corpus = np.asarray(corpus, dtype=dtype)

        self.tokens = tokens

        # cast values in `tokens` as numpy arrays
        for k,v in self.tokens.iteritems():
            self.tokens[k] = np.asarray(v)

        self.validate_tokens(self)

        self.terms = self.extract_terms(self.corpus,
                                        dtype=self.corpus.dtype)



    def __getitem__(self, i):

        return self.corpus[i]


    def view_tokens(self, name):
        """
        Displays a tokenization of the corpus.

        Parameters
        ----------
        name : an immutable
           The name of a tokenization in `tokens`.

        Returns
        -------
        A tokenized view of `corpus`.

        See Also
        --------
        BaseCorpus
        numpy.split

        """
        
        return np.split(self.corpus, self.tokens[name])


    @staticmethod
    def extract_terms(corpus, dtype=None):
        """
        Extracts the term set of a corpus.
        
        Parameters
        ----------
        corpus : array-like

        Returns
        -------
        An indexed set of the elements in the object as a 1-D array.
        """
        ind_term_set = list(set(corpus))

        return np.asarray(ind_term_set, dtype=dtype)


    @staticmethod
    def validate_tokens(bcorpus):
        """
        Checks for invalid tokenizations. Specifically, checks to see
        that the list of indices are sorted and are in range. Ignores
        empty tokens.

        Parameters
        ----------
        bcorpus : a BaseCorpus object

        Returns
        -------
        True if the tokenizations are all valid; otherwise raises an
        exception.

        Raises
        ------
        TODO 

        """
        if bcorpus.tokens:
            for k,v in bcorpus.tokens.iteritems():
                
                for i,j in enumerate(v):
                    
                    if i < len(v)-1 and j > v[i+1]:

                        msg = 'malsorted tokenization for ' + str(k)\
                              + ': tokens ' + str(j) + ' and ' + str(v[i+1])

                        raise Exception(msg)
                    
                    if j > len(bcorpus.corpus):
                        
                        print v[-30:]

                        msg = 'invalid tokenization for ' + str(k)\
                              + ': ' + str(j) + ' is out of range ('\
                              + str(len(bcorpus.corpus)) + ')'
                        
                        raise Exception(msg)

                    #TODO: Define a proper exception

        return True



    




class Corpus(BaseCorpus):
    """
    The goal of the Corpus class is to provide an efficient
    representation of a textual corpus.

    A Corpus object contains an integer representation of the text and
    maps to permit conversion between integer and string
    representations of a given term.

    As a subclass of BaseCorpus it includes a dictionary of
    tokenizations of the corpus and a method for viewing (without
    copying) these tokenizations.

    A Corpus object also stores metadata (e.g., document names)
    associated with the available tokenizations.

    Parameters
    ----------
    corpus : array-like
        A string array representing the corpus as a sequence of atomic
        terms.
    tokens : dict-like with 1-D integer array-like values, optional
        Each key in `tokens` is a name of a tokenization. Each value
        in `tokens` is an array containing the indices marking the
        token boundaries. A value in `tokens` is intended for use as a
        value for the `indices_or_sections` parameter in
        `numpy.split`. Default is `None`.
    tokens_meta : dict-like with 1-D array-like values, optional
        Each key in `tokens_meta` is a name of a tokenization. Each
        value in `tokens_meta` is an array containing metadata
        corresponding to a tokenization found in `tokens`. The
        tokenization metadata is checked on initialization by
        `_validate_metadata`.

    Attributes
    ----------
    corpus : 1-D 32-bit integer array
        corpus is the integer representation of the input string
        array-like value value of the corpus parameter
    terms : 1-D string array
        The indexed set of strings occurring in corpus. It is a
        string-typed array.
    terms_int : 1-D 32-bit integer array
        A dictionary whose keys are `terms` and whose values are their
        corresponding integers (i.e., indices in `terms`).
    tokens : dict with 1-D numpy integer arrays as values
        Stores the value of the `tokens` parameter. See `BaseCorpus`.
    tokens_meta : dict with 1-D arrays as values
        Stores the value of the `tokens_meta` parameter. Values are
        cast as 1-D arrays.
        
    Methods
    -------

    view_tokens
        Takes a name of tokenization and returns a view of the corpus
        tokenized accordingly. The optional parameter `strings` takes
        a boolean value: True to view string representations of terms;
        False to view integer representations of terms. Default is
        `False`.
    extract_terms
        Static method. Takes an array-like object and returns an
        indexed set of the elements in the object as a 1-D numpy
        array.
    validate_tokens
        Static method. Takes a BaseCorpus-like object and verifies
        that the tokenizations are sorted and in range.
    _validate_meta
        Verifies that the keys in `tokens_meta` are also in `tokens`;
        and that array lengths in `tokens_meta` do not exceed those in
        `tokens`. See the `tokens_meta` parameter.
    gen_lexicon
        Returns a copy of itself but with `corpus`, `tokens`, and
        `tokens_meta` set to None. Occasionally, the only information
        needed from the Corpus object is the mapping between string
        and integer representations of terms; this provides a smaller
        version of the corpus object for such situations.
    save
        Takes a filename and saves the data contained in a Corpus
        object to a `npy` file using `numpy.savez`.
    load
        Static method. Takes a filename, loads the file data into a
        Corpus object and returns the object
    
    See Also
    --------
    BaseCorpus

    Examples
    --------

    >>> text = ['I', 'came', 'I', 'saw', 'I', 'conquered']
    >>> sents = [2, 4]
    >>> meta = ['Veni', 'Vidi', 'Vici']
    >>> tokens = {'sentences': sents}
    >>> tokens_meta = {'sentences': meta}

    >>> from inphosemantics.corpus import Corpus
    >>> c = Corpus(text, tokens=tokens, tokens_meta=tokens_meta)
    >>> c.corpus
    array([0, 3, 0, 2, 0, 1], dtype=int32)
    
    >>> c.terms
    array(['I', 'conquered', 'saw', 'came'],
          dtype='|S9')

    >>> c.terms_int['saw']
    2

    >>> c.view_tokens('sentences')
    [array([0, 3], dtype=int32), array([0, 2], dtype=int32),
     array([0, 1], dtype=int32)]

    >>> c.view_tokens('sentences', True)
    [array(['I', 'came'],
          dtype='|S4'), array(['I', 'saw'],
          dtype='|S3'), array(['I', 'conquered'],
          dtype='|S9')]

    >>> c.tokens_meta['sentences'][1]
    'Vidi'
    
    """
    
    def __init__(self,
                 corpus,
                 tokens=None,
                 tokens_meta=None):

        super(Corpus, self).__init__(corpus, tokens=tokens)

        self.terms_int =\
            dict(zip(self.terms, xrange(len(self.terms))))
        
        self.corpus =\
            np.asarray([self.terms_int[term]
                        for term in self.corpus], dtype=np.int32)


        self.tokens_meta = tokens_meta
        
        # cast tokens_meta values as numpy arrays
        for k,v in self.tokens_meta.iteritems():
            self.tokens_meta[k] = np.asarray(v)

        self._validate_meta()




    def _validate_meta(self):
        """
        Verifies that the keys in `tokens_meta` are also in `tokens`;
        and that array lengths in `tokens_meta` do not exceed those in
        `tokens`.

        Parameters
        ----------
        None

        Returns
        -------
        True

        Raises
        ------
        TODO

        See Also
        --------
        Corpus.
        """
        #TODO proper exceptions and messages
        for k in self.tokens_meta:

            if k not in self.tokens:

                raise Exception(str(k) + ' is not a tokenization type')

            if self.tokens_meta[k].shape[0] > self.tokens[k].shape[0]:

                raise Exception('Metadata mismatch for ' + str(k))

        return True
            



    def view_tokens(self, name, strings=False):
        """
        Displays a tokenization of the corpus.

        Parameters
        ----------
        name : an immutable
           The name of a tokenization in `tokens`.
        strings : Boolean, optional
            If True, string representations of terms are returned.
            Otherwise, integer representations are returned. Default
            is `False`.

        Returns
        -------
        A tokenized view of `corpus`.

        See Also
        --------
        Corpus
        BaseCorpus
        """
        
        token_list = super(Corpus, self).view_tokens(name)

        if strings:

            for i,token in enumerate(token_list):

                token_str = [self.terms[t] for t in token]
                    
                token_list[i] = np.array(token_str, dtype=np.str_)
            
        return token_list




    def gen_lexicon(self):
        """
        Returns a copy of itself but with `corpus`, `tokens`, and
        `tokens_meta` set to None.
        
        See Corpus.
        """
        c = Corpus([])
        c.corpus = None
        c.tokens = None
        c.tokens_meta = None
        c.terms = self.terms
        c.terms_int = self.terms_int

        return c


    @staticmethd
    def load(filename):
        """
        Loads data into a Corpus object that has been stored using
        `save`.
        
        Parameters
        ----------
        filename : str-like or file-like object
            Designates the file to read. If `filename` is a string
            ending in `.gz`, the file is first gunzipped. See
            `numpy.load` for further details.

        Returns
        -------
        A Corpus object storing the data found in `filename`.

        See Also
        --------
        Corpus
        Corpus.save
        numpy.load
        """
        
        pass


    def save(filename):
        """
        Saves data from a Corpus object as an `npz` file.
        
        Parameters
        ----------
        filename : str-like or file-like object
            Designates the file to which to save data. See
            `numpy.savez` for further details.

        Returns
        -------
        None

        See Also
        --------
        Corpus
        Corpus.load
        numpy.savez
        """
        
        pass


    #Legacy
    def dump(self, filename, terms_only=False):

        if terms_only:

            self.gen_lexicon().dump(filename)

        else:
            super(Corpus, self).dump(filename)

    #Legacy
    def dumpz(self, filename, terms_only=False):

        if terms_only:

            self.gen_lexicon().dumpz(filename)

        else:
            super(Corpus, self).dumpz(filename)


