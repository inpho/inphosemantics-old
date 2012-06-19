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
        tokens) making up the corpus.
    tok_data : list-like with 1-D integer array-like elements, optional
        Each element in `tok_data` is an array containing the indices
        marking the token boundaries. An element in `tok_data` is
        intended for use as a value for the `indices_or_sections`
        parameter in `numpy.split`. Elements of `tok_data` may also be
        1-D arrays whose elements are pairs, where the first element
        is a token boundary and the second element is metadata
        associated with that token preceding that boundary. For
        example, (250, 'dogs') might indicate that the 'article' token
        ending at the 250th term of the corpus is named 'dogs'.
        Default is `None`.
    tok_names : array-like, optional
        Each element in `tok_names` is a name of a tokenization in
        `tok_data`.
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
    tok : dict with 1-D numpy arrays as values
        The tokenization dictionary. Stems of key names are given by
        `tok_names`. A key name whose value is the array of indices
        for a tokenization has the suffix '_indices'. A key name whose
        value is the metadata array for a tokenization has the suffix
        '_metadata'.

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
    >>> tok_names = ['sentences']
    >>> tok_data = [[(5, 'transitive'), (9, 'intransitive')]]

    >>> from inphosemantics.corpus import BaseCorpus
    >>> c = BaseCorpus(corpus, tok_names=tok_names, tok_data=tok_data)
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

    >>> c.view_metadata('sentences')[0]
    'transitive'

    """
    def __init__(self,
                 corpus,
                 dtype=None,
                 tok_names=None,
                 tok_data=None):

        self.corpus = np.asarray(corpus, dtype=dtype)

        self.terms = self.extract_terms(self.corpus,
                                        dtype=self.corpus.dtype)


        self.tok = dict()

        for i,t in enumerate(tok_data):
            
            try:
                # Suppose that `tok` is array-like with tuple-like
                # elements (i.e., it has metadata)
                indices, metadata = zip(*t)

                indices = np.asarray(indices)
                metadata = np.asarray(metadata)

                if self._validate_metadata(indices, metadata):
                    self.tok[tok_names[i] + '_metadata'] = metadata

            except TypeError:

                # Suppose instead that `tok` has unsubscriptable
                # elements (i.e., it has no metadata)
                indices = np.asarray(t)


            if self._validate_indices(indices):
                self.tok[tok_names[i] + '_indices'] = indices

                




    def __getitem__(self, i):

        return self.corpus[i]


    def _validate_indices(self, indices):
        """
        Checks for invalid tokenizations. Specifically, checks to see
        that the list of indices are sorted and are in range. Ignores
        empty tokens.

        Parameters
        ----------
        indices : 1-D integer array-like

        Returns
        -------
        True if the indices are validated

        Raises
        ------
        Exception

        See Also
        --------
        BaseCorpus
        """
        #TODO: Define a custom exception

        for i,j in enumerate(indices):
                    
            if i < len(indices)-1 and j > indices[i+1]:

                msg = 'malsorted tokenization:'\
                      ' tok ' + str(j) + ' and ' + str(indices[i+1])

                raise Exception(msg)
                    
            if j > self.corpus.shape[0]:

                print type(j)
                        
                print indices[-30:]

                msg = 'invalid tokenization'\
                      ' : ' + str(j) + ' is out of range ('\
                      + str(self.corpus.shape[0]) + ')'
                
                raise Exception(msg)

        return True


    @staticmethod
    def _validate_metadata(indices, metadata):
        """
        Verifies that if there is metadata, there is an item of
        metadata for and only for each index.

        Parameters
        ----------
        indices : 1-D integer array-like

        metadata : 1-D array-like

        Returns
        -------
        True if metadata is validated.

        Raises
        ------
        Exception

        See Also
        --------
        BaseCorpus
        """
        if len(indices) != len(metadata):

            msg = 'Mismatch between indices and metadata:\n'\
                  '  ' + str(len(indices)) + ' indices\n'\
                  '  ' + str(len(metadata)) + ' metadata'
            
            raise Exception(msg)

        return True


    def view_tokens(self, name):
        """
        Displays a tokenization of the corpus.

        Parameters
        ----------
        name : string-like
           The name of a tokenization.

        Returns
        -------
        A tokenized view of `corpus`.

        See Also
        --------
        BaseCorpus
        numpy.split

        """

        k = name + '_indices'

        tokens = np.split(self.corpus, self.tok[k])

        # If the final token break is the end of the corpus, remove
        # the trailing empty array produced by np.split
        if self.tok[k][-1] == len(self.corpus):
            del tokens[-1]
                
        return tokens



    def view_metadata(self, name):
        """
        Displays the metadata corresponding to a tokenization of the
        corpus.

        Parameters
        ----------
        name : string-like
            The name of a tokenization.

        Returns
        -------
        The metadata for a tokenization.

        See Also
        --------
        BaseCorpus
        """
        k = name + '_metadata'
        return self.tok[k]
    



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

        See Also
        --------
        BaseCorpus
        """
        ind_term_set = list(set(corpus))

        return np.asarray(ind_term_set, dtype=dtype)



    




class Corpus(BaseCorpus):
    """
    The goal of the Corpus class is to provide an efficient
    representation of a textual corpus.

    A Corpus object contains an integer representation of the text and
    maps to permit conversion between integer and string
    representations of a given term.

    As a BaseCorpus object, it includes a dictionary of tokenizations
    of the corpus and a method for viewing (without copying) these
    tokenizations. This dictionary also stores metadata (e.g.,
    document names) associated with the available tokenizations.

    Parameters
    ----------
    corpus : array-like
        A string array representing the corpus as a sequence of atomic
        terms.
    tok_data : list-like with 1-D integer array-like elements, optional
        Each element in `tok_data` is an array containing the indices
        marking the token boundaries. An element in `tok_data` is
        intended for use as a value for the `indices_or_sections`
        parameter in `numpy.split`. Elements of `tok_data` may also be
        1-D arrays whose elements are pairs, where the first element
        is a token boundary and the second element is metadata
        associated with that token preceding that boundary. For
        example, (250, 'dogs') might indicate that the 'article' token
        ending at the 250th term of the corpus is named 'dogs'.
        Default is `None`.
    tok_names : array-like, optional
        Each element in `tok_names` is a name of a tokenization in
        `tok_data`.

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
    tok : dict with 1-D numpy arrays as values
        The tokenization dictionary. Stems of key names are given by
        `tok_names`. A key name whose value is the array of indices
        for a tokenization has the suffix '_indices'. A key name whose
        value is the metadata array for a tokenization has the suffix
        '_metadata'.
        
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
    >>> tok_names = ['sentences']
    >>> tok_data = [[(2, 'Veni'), (4, 'Vidi'), (6, 'Vici')]]

    >>> from inphosemantics.corpus import Corpus
    >>> c = Corpus(text, tok_names=tok_names, tok_data=tok_data)
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

    >>> c.view_tokens('sentences', strings=True)
    [array(['I', 'came'],
          dtype='|S4'), array(['I', 'saw'],
          dtype='|S3'), array(['I', 'conquered'],
          dtype='|S9')]

    >>> c.view_metadata('sentences')[1]
    'Vidi'
    
    """
    
    def __init__(self,
                 corpus,
                 tok_names=None,
                 tok_data=None):

        super(Corpus, self).__init__(corpus,
                                     tok_names=tok_names,
                                     tok_data=tok_data)

        self._set_terms_int()

        # Integer encoding of a string-type corpus
        self.corpus = np.asarray([self.terms_int[term]
                                  for term in self.corpus],
                                 dtype=np.int32)


    def _set_terms_int(self):
        """
        Maps elements of a string array to their indices.
        """
        self.terms_int = dict(zip(self.terms,
                                  xrange(len(self.terms))))




    def view_tokens(self, name, strings=False):
        """
        Displays a tokenization of the corpus.

        Parameters
        ----------
        name : string-like
           The name of a tokenization.
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


    @staticmethod
    def load(file):
        """
        Loads data into a Corpus object that has been stored using
        `save`.
        
        Parameters
        ----------
        file : str-like or file-like object
            Designates the file to read. If `file` is a string ending
            in `.gz`, the file is first gunzipped. See `numpy.load`
            for further details.

        Returns
        -------
        A Corpus object storing the data found in `file`.

        See Also
        --------
        Corpus
        Corpus.save
        numpy.load
        """

        arrays_in = np.load(file)

        c = Corpus([])
        c.corpus = arrays_in['corpus']
        c.terms = arrays_in['terms']
        c._set_terms_int()

        for k in arrays_in:
            if k.endswith('_indices') or k.endswith('_metadata'):
                c.tok[k] = arrays_in[k]

        return c



    def save(file):
        """
        Saves data from a Corpus object as an `npz` file.
        
        Parameters
        ----------
        file : str-like or file-like object
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

        # Arrays to save: corpus, terms, values of tok

        arrays_out = dict(**self.tok)
        arrays_out['corpus'] = self.corpus
        arrays_out['terms'] = self.terms

        np.savez(file, **arrays_out)



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



class MaskedCorpus(Corpus):

    def __init__(self,
                 corpus,
                 tok_names=None,
                 tok_data=None,
                 mask_terms=None,
                 fill_value=None):

        super(MaskedCorpus, self).__init__(corpus,
                                           tok_names=tok_names,
                                           tok_data=tok_data)

        self.corpus = np.ma.MaskedArray(self.corpus)

        self.mask_terms = np.asarray(mask_terms)

        # Encode mask_terms as integers
        self.mask_terms_int = dict([(t, self.terms_int[t])
                                       for t in mask_terms])

        mt_vals = self.mask_terms_int.values()

        # nditer doesn't facilitate (in an obvious way) updating the
        # mask; so...
        for i,term in enumerate(self.corpus):
            if term in mt_vals:
                self.corpus[i] = np.ma.masked



    @staticmethod
    def load(file):
        pass


    def save(file):
        pass
        






def test():

    text = ['I', 'came', 'I', 'saw', 'I', 'conquered']
    tok_names = ['sentences']
    tok_data = [[(2, 'Veni'), (4, 'Vidi'), (6, 'Vici')]]
    mask_terms = ['I']

    c = MaskedCorpus(text,
                     tok_names=tok_names,
                     tok_data=tok_data,
                     mask_terms=mask_terms)
    
    return c.corpus
