import bz2
import pickle

import numpy as np


class Corpus(object):
    """
    term_types is the *set* of term tokens recast as a list (so an
    indexed set).

    'partitions' should be a dictionary whose values are lists of
    indices. They are verified presently at initialization.

    """
    def __init__(self, term_tokens, partitions=None, dtype=None,
                 stoplist=None, filename=None):

        self.term_tokens = np.asarray(term_tokens, dtype=dtype)
        self.stoplist = stoplist

        # Verify valid partitions
        if partitions:
            for k,v in partitions.iteritems():
                
                #Allows empty partitions
                for i,j in enumerate(v):
                    
                    # checking to see that it is sorted and that the
                    # indices are in range
                    if ((i < len(v)-1 and j > v[i+1]) 
                        or j >= len(self.term_tokens)):
                        
                        #TODO: Define a proper exception
                        raise Exception('invalid partitioning', k, v)

        self.partitions = partitions

        self._set_term_types()


    def __getitem__(self, i):
        return self.term_tokens[i]

    def _set_term_types(self):
        
        self.term_types = list(set(self.term_tokens))


    def view_partition(self, name, decoder=None):
        parted = np.split(self.term_tokens, self.partitions[name])

        if decoder:
            return [ decoder.convert(part) for part in parted ]
        else:
            return parted

    
    def dump(self):

        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)


    def dumpz(self):

        f = bz2.BZ2File(self.filename + '.bz2', 'w')
        try:
            f.write(pickle.dumps(self))
        finally:
            f.close()


    def digitize(self):

        print 'Getting word types'
        words = self.term_types
        word_dict = dict(zip(words, xrange(len(words))))
        
        print 'Extracting sequence of word tokens'
        int_corp = [word_dict[token] for token in self.term_tokens]

        digitizedCorpus = Corpus(int_corp, partitions=self.partitions, dtype=np.uint32)
        decoder = CorpusDecoder( self.term_types )
    
        return digitizedCorpus,decoder









class CorpusDecoder(object):

    def __init__(self, term_types): # possibly also attach partitions
        self.term_types  = term_types

    def decode(self, token):
        return self.term_types[token]
    
    def convert(self, term_tokens):
        return [ self.term_types[token] for token in term_tokens ]



def load_picklez(filename):
    """
    Takes a filename and loads it as though it were a python pickle.
    If the file ends in '.bz2', tries to decompress it first.
    """
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

#A stop-gap measure to generate new style Corpus instances from old
def import_corpus(corpus, param):
    
    from inphosemantics.corpus import Corpus as CorpusOld

    corpus = CorpusOld(corpus, param)

    page_names = os.listdir(corpus.tokenized_path)
    
    pages = [corpus.tokenized_document(name) 
             for name in page_names]

    return Corpus(pages, stopwords=corpus.stopwords)



class SepPartitions(object):

    # self.term_tokens ?

    def articles(self, path):

        article_names = os.listdir(path)

        partition = []
        
        for name in article_names:
            # the length of the articles give the partitions

            pass
        
        return


    def paragraphs(self, path):

        pass


    def sentences(self, path):

        pass



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






############################
####      TEST DATA    #####
############################

def genCorpus():
    return Corpus(['cats','chase','dogs','dogs','do','not','chase','cats'], {'sentences' : [3]})
