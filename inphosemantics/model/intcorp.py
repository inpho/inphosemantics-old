import bz2
import pickle

import numpy as np


class Corpus(object):
    """
    term_types is the *set* of term tokens recast as a list (so an
    indexed set).

    'token_dict' should be a dictionary whose values are lists of
    indices. They are verified presently at initialization.

    """
    def __init__(self, term_tokens, token_dict=None, dtype=None,
                 stoplist=None, filename=None):

        self.term_tokens = np.asarray(term_tokens, dtype=dtype)
        self.stoplist = stoplist

        self.token_dict = token_dict
        self.validate_token_dict()

        self._set_term_types()


    def __getitem__(self, i):
        return self.term_tokens[i]

    def _set_term_types(self):
        
        self.term_types = list(set(self.term_tokens))


    def view_tokens(self, name, decoder=None):

        #TODO: A user might reasonably try to view 'words' or 'terms'.
        #Handle this.
        
        tokens = np.split(self.term_tokens, self.token_dict[name])

        if decoder:
            return [ decoder.convert(token) for token in tokens ]
        else:
            return tokens

    
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

        digitizedCorpus = Corpus(int_corp, token_dict=self.token_dict, dtype=np.uint32)
        decoder = CorpusDecoder( self.term_types )
    
        return digitizedCorpus,decoder


    def validate_token_dict(self):
        """
        Checks for invalid tokenizations. Specifically, checks to see
        that the list of indices are sorted and are
        in range. Allows empty tokens.
        """
        if self.token_dict:
            for k,v in self.token_dict.iteritems():
                
                for i,j in enumerate(v):
                    
                    if ((i < len(v)-1 and j > v[i+1]) 
                        or j >= len(self.term_tokens)):
                        
                        #TODO: Define a proper exception
                        raise Exception('invalid tokenization', k, v)

        return True








class CorpusDecoder(object):

    def __init__(self, term_types): # possibly also attach tokens
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



from inphosemantics.corpus.tokenizer import *


class SepTokens(object):

    def __init__(self, path):

        self.path = path
        self.words = []
        self.article_meta = {}
        self.articles, self.paragraphs, self.sentences =\
            self._compute_tokens()


    def _compute_tokens(self):

        articles, metadata = textfile_tokenize(self.path)

        self.article_meta = metadata

        article_tokens = []
        paragraph_tokens = []
        sentence_spans = []

        for i,article in enumerate(articles):

            print 'Processing article in', self.article_meta[i]

            paragraphs = paragraph_tokenize(article)
            
            for paragraph in paragraphs:
                sentences = sentence_tokenize(paragraph)

                for sentence in sentences:
                    words = word_tokenize(sentence)

                    self.words.extend(words)
                    
                    sentence_spans.append(len(words))

                paragraph_tokens.append(sum(sentence_spans))
                    
            article_tokens.append(sum(sentence_spans))


        sentence_tokens =\
            [sum(sentence_spans[:i+1])
             for i in xrange(len(sentence_spans) - 1)]


        article_tokens = article_tokens[:-1]
        paragraph_tokens = paragraph_tokens[:-1]
        sentence_tokens = sentence_tokens[:-1]


        return article_tokens, paragraph_tokens,\
               sentence_tokens


    @property
    def tokens_dict(self):

        d = dict(articles = self.articles,
                 paragraphs = self.paragraphs,
                 sentences = self.sentences)

        return d


class IepTokens(SepTokens):
    pass








############################
####      TEST DATA    #####
############################

def genCorpus():
    return Corpus(['cats','chase','dogs','dogs','do','not','chase','cats'], {'sentences' : [3]})


def test_IepTokens():

    path = 'test-data/iep-selected'

    parts = IepTokens(path)

    print 'Article breaks:\n', parts.articles
    print '\nParagraph breaks:\n', parts.paragraphs
    print '\nSentence breaks:\n', parts.sentences

    c = Corpus(parts.words, parts.tokens_dict)

    return c

