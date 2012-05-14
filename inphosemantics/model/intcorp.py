import bz2
import pickle

import numpy as np

class IntegerCorpus(object):
    """
    """
    def __init__(self, int_list, dtype=np.uint32, 
                 partitions=None, filename=None):
        
        self.int_array = np.asarray(int_list, dtype=dtype)

        if self.int_array.ndim > 1:
            raise ValueError('integer corpus must be 0 or 1-dim')

        #TODO: This verification needs to take place whenever
        #self.partitions is updated

        # Verify valid partitions
        if partitions:
            for k,v in partitions.iteritems():
                
                #Allows empty partitions
                for i,j in enumerate(v):
                    
                    # checking to see that it is sorted and that the
                    # indices are in range
                    if ((i < len(v)-1 and j > v[i+1]) 
                        or j >= len(self.array_list)):
                        
                        #TODO: Define a proper exception
                        raise Exception('invalid partitioning', k, v)

        self.partitions = partitions
        self.filename = filename


    def view_partition(self, name):

        return [np.asarray(sub) for sub 
                in np.split(self.int_array, self.partitions[name])]

    
    def dump(self):

        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)


    def dumpz(self):

        f = bz2.BZ2File(self.filename + '.bz2', 'w')
        try:
            f.write(pickle.dumps(self))
        finally:
            f.close()



def load_intcorp(filename):
    
    if filename.endswith('.bz2'):

        f = bz2.BZ2File(filename, 'r')
        try:
            return pickle.loads(f.read())
        finally:
            f.close()
    
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)



import os


class Corpus(object):
    """
    term_types is the *set* of term tokens recast as a list (so an
    indexed set).

    'partitions' should be a dictionary whose values are lists of
    indices. They are verified presently at initialization.

    """
    def __init__(self, term_tokens, partitions=None, 
                 stoplist=None, filename=None):

        self.term_tokens = term_tokens
        self.stoplist = stoplist

        #TODO: This verification needs to take place whenever
        #self.partitions is updated

        # Verify valid partitions
        if partitions:
            for k,v in partitions.iteritems():
                
                #Allows empty partitions
                for i,j in enumerate(v):
                    
                    # checking to see that it is sorted and that the
                    # indices are in range
                    if ((i < len(v)-1 and j > v[i+1]) 
                        or j >= len(obj)):
                        
                        #TODO: Define a proper exception
                        raise Exception('invalid partitioning', k, v)

        self.partitions = partitions

        self._set_term_types()


    def __getitem__(self, i):
        return self.term_tokens[i]

    #TODO: update term_types whenever self.term_tokens is updated. Or
    #make self.term_tokens immutable.
    def _set_term_types(self):
        
        self.term_types = list(set(self.term_tokens))

    

#A stop-gap measure to generate new style Corpus instances from old
def load_corpus(corpus, param):
    
    from inphosemantics.corpus import Corpus as CorpusOld

    corpus = CorpusOld(corpus, param)

    page_names = os.listdir(corpus.tokenized_path)
    
    pages = [corpus.tokenized_document(name) 
             for name in page_names]

    return Corpus(pages, stopwords=corpus.stopwords)



def digitize_corpus(corpus):
    """
    corpus is a Corpus instance from inphosemantics.corpus
    """
    # Underlying integer sequence
    print 'Getting word types'
    words = corpus.word_types
    word_dict = dict(zip(words, xrange(len(words))))
    
    print 'Extracting sequence of word tokens'
    tokens = flatten(corpus.pages)
    int_corp = [word_dict[token] for token in tokens]

    # Partitions
    l = ind_flatten(corpus.pages)
    #TODO: Write a function to generalize all of this zipping

    print 'Determining page partitions'
    pages_partition = delta(zip(*l)[0])

    print 'Determining paragraph partitions'
    paragraphs_partition = delta(zip(zip(*l)[0], zip(*l)[1]))
    
    print 'Determining sentence partitions'
    sentences_partition = delta(zip(zip(*l)[0], zip(*l)[1], zip(*l)[2]))

    partitions = dict(pages = pages_partition,
                      paragraphs = paragraphs_partition,
                      sentences = sentences_partition)

    return IntegerCorpus(int_corp, partitions=partitions)



def delta(ls):
    """
    Takes a list-like object representing a sequence of integers or
    tuples of integers and returns the indices where the integer or
    tuple at i != that at i+1
    """
    result = []

    for i in xrange(len(ls)-2):
        if ls[i] != ls[i+1]:
            result.append(i+1)

    return result
        

    
def flatten(ls):
    """
    Flattens a list of lists. NB: not at all general; reaches maximum
    recursion depth quickly.
    """
    def f(ls):
        result = []

        for x in ls:
            if type(x) == list:
                result.extend(x)
            else:
                result.append(x)
                
        return result
    
    if type(ls) == list:
        return f(map(flatten, ls))
    else:
        return ls



def ind_flatten(ls):
    """
    Flattens a list of lists. NB: not at all general; reaches maximum
    recursion depth quickly.
    """
    def f(ls):
        result = []

        for i,x in enumerate(ls):
            if type(x) == list:
                result.extend([flatten([i,y]) for y in x])
            else:
                result.append(x)
                
        return result
    
    if type(ls) == list:
        return f(map(ind_flatten, ls))
    else:
        return ls





# class Corpus(object):
#     """
#     pages is a list of pages. Each page is a list of paragraphs. Each
#     paragraph is a list of sentences. Each sentence is a list of
#     word tokens. Each word token is a string.

#     word_types is the *set* of word tokens recast as a list to give an
#     indexing.

#     """
#     def __init__(self, pages, stopwords=None):

#         self.pages = pages
#         self.stopwords = stopwords
#         self._set_word_types()


#     #TODO: update word_types whenever pages is updated. Or make pages
#     #immutable.
#     def _set_word_types(self):
        
#         self.word_types = list(set(flatten(self.pages)))


# def test_corpus():
    
#     c = Corpus(
#         [[[[u'zombies', u'are', u'exactly', u'like', u'us', u'in', u'all',
#             u'physical', u'respects', u'but', u'have', u'no', u'conscious',
#             u'experiences', u'by', u'definition', u'there', u'is', u'nothing',
#             u'it', u'is', u'like', u'to', u'be', u'a', u'zombie'],
#            [u'yet', u'zombies', u'behave', u'like', u'us', u'and', u'some',
#             u'even', u'spend', u'a', u'lot', u'of', u'time', u'discussing',
#             u'consciousness'],
#            [u'this', u'disconcerting', u'fantasy', u'helps', u'to', u'make',
#             u'the', u'problem', u'of', u'phenomenal', u'consciousness',
#             u'vivid', u'especially', u'as', u'a', u'problem', u'for',
#             u'physicalism']],
#           [[u'few', u'people', u'think', u'zombies', u'actually', u'exist'],
#            [u'but', u'many', u'hold', u'they', u'are', u'at', u'least',
#             u'conceivable', u'and', u'some', u'that', u'they', u'are',
#             u'logically', u'or', u'metaphysically', u'possible'], 
#            [u'it', u'is', u'argued', u'that', u'if', u'zombies', u'are',
#             u'so', u'much', u'as', u'a', u'bare', u'possibility', u'then',
#             u'physicalism', u'is', u'false', u'and', u'some', u'kind',
#             u'of', u'dualism', u'must', u'be', u'accepted'],
#            [u'for', u'many', u'philosophers', u'that', u'is', u'the',
#             u'chief', u'importance', u'of', u'the', u'zombie', u'idea'],
#            [u'but', u'the', u'idea', u'is', u'also', u'of', u'interest',
#             u'for', u'its', u'presuppositions', u'about', u'the', u'nature',
#             u'of', u'consciousness', u'and', u'how', u'the', u'physical',
#             u'and', u'the', u'phenomenal', u'are', u'related'],
#            [u'use', u'of', u'the', u'zombie', u'idea', u'against',
#             u'physicalism', u'also', u'raises', u'more', u'general',
#             u'questions', u'about', u'relations', u'between',
#             u'imaginability', u'conceivability', u'and', u'possibility'], 
#            [u'finally', u'zombies', u'raise', u'epistemological',
#             u'difficulties', u'they', u'reinstate', u'the', u'other',
#             u'minds', u'problem']]]])

#     d = digitize_corpus(c)

#     print 'Sentence view is lossless:',\
#         (np.asarray(d) == np.hstack(d.view_partition('sentences'))).all()

#     print 'Paragraph view is lossless:',\
#         (np.asarray(d) == np.hstack(d.view_partition('paragraphs'))).all()

#     print 'Pages view is lossless:',\
#         (np.asarray(d) == np.hstack(d.view_partition('pages'))).all()


#     return c,d


# def digitize_sep():

#     c = load_corpus('sep','complete')
#     return digitize_corpus(c)


# class IntegerCorpus(np.ndarray):
#     """
#     For an explanation of the mechanics of this subclass, see how to
#     subclass numpy ndarrays:
#     http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing
#     """

#     #TODO: More thought needs to be put into what happens to the
#     #partitions when IntegerCorpus objects are manipulated and combined.

#     def __new__(cls, array, dtype=np.uint32, partitions=None, filename=None):

#         obj = np.asarray(array, dtype=dtype).view(cls)

#         if obj.ndim > 1:
#             raise ValueError('integer corpus must be 0 or 1-dim')

#         #TODO: This verification needs to take place whenever
#         #self.partitions is updated

#         # Verify valid partitions
#         if partitions:
#             for k,v in partitions.iteritems():
                
#                 #Allows empty partitions
#                 for i,j in enumerate(v):
                    
#                     # checking to see that it is sorted and that the
#                     # indices are in range
#                     if ((i < len(v)-1 and j > v[i+1]) 
#                         or j >= len(obj)):
                        
#                         #TODO: Define a proper exception
#                         raise Exception('invalid partitioning', k, v)

#         obj.partitions = partitions
#         obj.filename = filename
        
#         return obj



#     def __array_finalize__(self, obj):

#         if obj is None:
#             return

#         self.partitions = getattr(obj, 'partitions', None)
#         self.filename = getattr(obj, 'filename', None)



#     def view_partition(self, name):

#         return [np.asarray(sub) for sub 
#                 in np.split(self, self.partitions[name])]

    
#     def dump(self):

#         with open(self.filename, 'wb') as f:
#             pickle.dump(self, f)


#     def dumpz(self):

#         f = bz2.BZ2File(self.filename + '.bz2', 'w')
#         try:
#             f.write(pickle.dumps(self))
#         finally:
#             f.close()
