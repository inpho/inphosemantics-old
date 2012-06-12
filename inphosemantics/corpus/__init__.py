import bz2
import pickle

import numpy as np


def extract_terms(corpus, dtype=None):

    return np.asarray(list(set(corpus)), dtype=dtype)


class BaseCorpus(object):
    """
    corpus should be list-like with elements of one type. On
    initialization, corpus will be converted to a numpy array.

    terms is the *set* of term tokens recast as a list (effectively an
    indexed set).

    'tokens' should be a dictionary whose values are lists of
    indices. They are verified presently at initialization.

    """
    def __init__(self, corpus, tokens=None, dtype=None):

        self.corpus = np.asarray(corpus, dtype=dtype)


        self.tokens = tokens

        # recast tokens_meta values as numpy arrays
        for k,v in self.tokens.iteritems():
            self.tokens[k] = np.asarray(v)

        self.validate_tokens()


        self.terms = extract_terms(corpus, dtype=dtype)



    def __getitem__(self, i):

        return self.corpus[i]



    def view_tokens(self, name):
        """
        Takes a key name.

        Returns a list of numpy arrays or

        if name == 'terms' and 'terms' is not a key in tokens, returns corpus

        if name == 'words' and 'words' is not a key in tokens, returns corpus

        Intended usage: the key name is the name of a tokenization
        stored in tokens and the output is the actual list of tokens.

        """
        
        if ((name == 'terms' or name == 'words')
            and (self.tokens == None
                 or name not in self.tokens)):

            return self.corpus

        else:
            return np.split(self.corpus, self.tokens[name])




    def validate_tokens(self):
        """
        Checks for invalid tokenizations. Specifically, checks to see
        that the list of indices are sorted and are
        in range. Ignores empty tokens.
        """
        if self.tokens:
            for k,v in self.tokens.iteritems():
                
                for i,j in enumerate(v):
                    
                    if i < len(v)-1 and j > v[i+1]: 
                        raise Exception('malsorted tokenization for ' + str(k)
                                        + ': tokens ' + str(j)
                                        + ' and ' + str(v[i+1]))
                    
                    if j > len(self.corpus):
                        print v[-30:]
                        raise Exception('invalid tokenization for ' + str(k)
                                        + ': ' + str(j) + ' is out of range ('
                                        + str(len(self.corpus)) + ')')

                    #TODO: Define a proper exception

        return True




    def dump(self, filename):

        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    def dumpz(self, filename):

        f = bz2.BZ2File(filename, 'w')
        try:
            f.write(pickle.dumps(self))
        finally:
            f.close()

    




class Corpus(BaseCorpus):
    """
    The goal of the Corpus class is to provide an efficient
    representation of a textual corpus.

    A Corpus object contains an integer representation of the text and
    mapping objects to facilitate conversion between integer and
    string representations of the text.

    As a subclass of BaseCorpus it includes a dictionary of
    tokenizations of the corpus and a method for viewing (without
    copying) these tokenizations.

    A Corpus object also stores metadata (e.g., document names)
    associated with the available tokenizations.

    Expected usage. corpus would be the atomic tokens (e.g.,
    words) as strings. Instantiation of Corpus will map the
    corpus to integer representations of the terms.

    corpus is a representation of the input corpus as a numpy array of
    32-bit integer type.

    terms is the indexed set of strings occurring in corpus. It is a
    string-typed numpy array.

    terms_int is a mapping object whose keys are given by terms and
    whose values are their corresponding integers.


    Examples
    --------

    >>> text = ['I', 'came', 'I', 'saw', 'I', 'conquered']
    >>> sents = [2,4]
    >>> meta = ['Veni','Vidi','Vici']
    >>> tokens = {'sentences': sents}
    >>> tokens_meta = {'sentences': meta}

    >>> from inphosemantics.corpus import Corpus
    >>> c = Corpus(text, tokens, tokens_meta)
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
        
        # recast tokens_meta values as numpy arrays
        for k,v in self.tokens_meta.iteritems():
            self.tokens_meta[k] = np.asarray(v)





    def view_tokens(self, name, strings=False):
        """
        Extends BaseCorpus.view_tokens

        If strings == True, the terms are returned as their string
        representations.

        If strings == False, the terms are returned as their integer
        representations.
        """
        
        token_list = super(Corpus, self).view_tokens(name)

        if len(token_list) > 0:

            if strings:

                if np.isscalar(token_list[0]):

                    token_list = [self.terms[t] for t in token_list]

                    token_list = np.array(token_list, dtype=np.str_)

                else:
                    
                    for i,token in enumerate(token_list):

                        token_str = [self.terms[t] for t in token]
                    
                        token_list[i] = np.array(token_str, dtype=np.str_)
            

        return token_list






    def gen_lexicon(self):
        """
        Create a corpus object that contains only the terms (as
        integers and as strings) but not the corpus itself
        """
        c = Corpus([])
        c.terms = self.terms
        c.terms_int = self.terms_int

        return c
    

    def dump(self, filename, terms_only=False):

        if terms_only:

            self.gen_lexicon().dump(filename)

        else:
            super(Corpus, self).dump(filename)


    def dumpz(self, filename, terms_only=False):

        if terms_only:

            self.gen_lexicon().dumpz(filename)

        else:
            super(Corpus, self).dumpz(filename)




############################
####      TEST DATA    #####
############################

def genBaseCorpus():
    return BaseCorpus(['cats','chase','dogs','dogs',
                       'do','not','chase','cats'],
                      {'sentences' : [3]})


# def test_BaseCorpus():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/selected/corpus/plain'

#     tokens = IepTokens(path)

#     c = BaseCorpus(tokens.word_tokens, tokens.tokens)

#     print 'First article:\n', c.view_tokens('articles')[1]
#     print '\nFirst five paragraphs:\n', c.view_tokens('paragraphs')[:5]
#     print '\nFirst ten sentences:\n', c.view_tokens('sentences')[:10]

#     print '\nLast article:\n', c.view_tokens('articles')[-1]
#     print '\nLast five paragraphs:\n', c.view_tokens('paragraphs')[-5:]
#     print '\nLast ten sentences:\n', c.view_tokens('sentences')[-10:]

#     return c


# def test_integer_corpus():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/selected/corpus/plain'

#     tokens = IepTokens(path)

#     c = BaseCorpus(tokens.word_tokens, tokens.tokens)

#     int_corpus, encoder = c.encode_corpus()

#     print 'First article:\n',\
#           int_corpus.view_tokens('articles', encoder)[1]
#     print '\nFirst five paragraphs:\n',\
#           int_corpus.view_tokens('paragraphs', encoder)[:5]
#     print '\nFirst ten sentences:\n',\
#           int_corpus.view_tokens('sentences', encoder)[:10]

#     print '\nLast article:\n',\
#           int_corpus.view_tokens('articles', encoder)[-1]
#     print '\nLast five paragraphs:\n',\
#           int_corpus.view_tokens('paragraphs', encoder)[-5:]
#     print '\nLast ten sentences:\n',\
#           int_corpus.view_tokens('sentences', encoder)[-10:]

#     return int_corpus, encoder


# def test_Corpus():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/selected/corpus/plain'

#     tokens = IepTokens(path)

#     c = Corpus(tokens.word_tokens, tokens.tokens,
#                tokens.tokens_metadata)

#     print 'First article:\n',\
#           c.view_tokens('articles', True)[1]
#     print '\nFirst five paragraphs:\n',\
#           c.view_tokens('paragraphs', True)[:5]
#     print '\nFirst ten sentences:\n',\
#           c.view_tokens('sentences', True)[:10]

#     print '\nLast article:\n',\
#           c.view_tokens('articles', True)[-1]
#     print '\nLast five paragraphs:\n',\
#           c.view_tokens('paragraphs', True)[-5:]
#     print '\nLast ten sentences:\n',\
#           c.view_tokens('sentences', True)[-10:]

#     print '\nSource of second article:',\
#           c.view_metadata('articles')[2]

#     return c


# def test_Corpus_dumpz():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/selected/corpus/plain'
#     filename = 'test-data/iep/selected/corpus/iep-selected.pickle.bz2'

#     tokens = IepTokens(path)

#     c = Corpus(tokens.word_tokens, tokens.tokens,
#                tokens.tokens_metadata)

#     c.dumpz(filename)


# def test_Corpus_dumpz_plato():

#     from inphosemantics.corpus.tokenizer import IepTokens

#     path = 'test-data/iep/plato/corpus/plain'
#     filename = 'test-data/iep/plato/corpus/iep-plato.pickle.bz2'

#     tokens = IepTokens(path)

#     c = Corpus(tokens.word_tokens, tokens.tokens,
#                tokens.tokens_metadata)

#     c.dumpz(filename)



