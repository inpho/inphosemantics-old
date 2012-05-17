import bz2
import pickle

import numpy as np


class BaseCorpus(object):
    """
    term_types is the *set* of term tokens recast as a list (so an
    indexed set).

    'tokens_dict' should be a dictionary whose values are lists of
    indices. They are verified presently at initialization.

    """
    def __init__(self, term_tokens, tokens_dict=None, dtype=None,
                 filename=None):

        self.term_tokens = np.asarray(term_tokens, dtype=dtype)

        self.tokens_dict = tokens_dict
        self.validate_tokens_dict()

        self._set_term_types()


    def __getitem__(self, i):
        return self.term_tokens[i]

    def _set_term_types(self):
        
        self.term_types = list(set(self.term_tokens))


    def view_tokens(self, name, decoder=None):
        """
        Takes a key name and returns a list of lists of strings.
        Intended usage: the key name is the name of a tokenization in
        term_dict and the output is the actual list of tokens.

        'decoder' is an indexable mapping the term_tokens to
        something. Typical usage: where the term_tokens are integers,
        the decoder might be the term_types so that the output is a
        list of lists of strings.
        """
        
        #TODO: A user might reasonably try to view 'words' or 'terms'.
        #Handle this.
        
        tokens = np.split(self.term_tokens, self.tokens_dict[name])

        #TODO: Rewrite this so as to return a numpy array (i.e., an object
        #with a datatype)

        if decoder:
            return map(lambda l: [decoder[x] for x in l], tokens)
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

        digitized_corpus = BaseCorpus(int_corp, tokens_dict=self.tokens_dict, dtype=np.uint32)
    
        return digitized_corpus, self.term_types


    def validate_tokens_dict(self):
        """
        Checks for invalid tokenizations. Specifically, checks to see
        that the list of indices are sorted and are
        in range. Allows empty tokens.
        """
        if self.tokens_dict:
            for k,v in self.tokens_dict.iteritems():
                
                for i,j in enumerate(v):
                    
                    if ((i < len(v)-1 and j > v[i+1]) 
                        or j >= len(self.term_tokens)):
                        
                        #TODO: Define a proper exception
                        raise Exception('invalid tokenization', k, v)

        return True



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



class Corpus(BaseCorpus):
    """
    Corpus is an instance of BaseCorpus with datatype uint32 that
    bundles various useful data and methods associated with a given
    instance of BaseCorpus.

    filename

    stoplist

    metadata associated with any given tokenization: terms, sentences,
    etc.

    Expected usage. term_tokens would be the atomic tokens (e.g.,
    words) as strings. Instantiation of Corpus will digitize
    term_tokens and store term_types as a list of strings under
    self.term_types_str.
    
    """
    
    def __init__(self, term_tokens, tokens_dict=None, tokens_meta=None,
                 filename=None, stoplist=None):

        BaseCorpus.__init__(self, term_tokens, tokens_dict=tokens_dict,
                            filename=filename)
        
        int_corp = self.digitize()

        self.term_tokens = int_corp[0].term_tokens
        # just the list of integers 0, ..., n
        self.term_types = int_corp[0].term_types 
        self.term_types_str = int_corp[1]

        self.stoplist = stoplist
        self.tokens_meta = tokens_meta


    def view_tokens(self, name, strings=False):

        if strings:
            return BaseCorpus.view_tokens(self, name, self.term_types_str)

        return BaseCorpus.view_tokens(self, name)


    def view_metadata(self, name):

        return self.tokens_meta[name]
        



from inphosemantics.corpus.tokenizer import *


class SepTokens(object):

    def __init__(self, path):

        self.path = path
        self.word_tokens = []
        self.articles_meta = None
        self.articles, self.paragraphs, self.sentences =\
            self._compute_tokens()


    def _compute_tokens(self):

        articles, metadata = textfile_tokenize(self.path)

        self.articles_meta = metadata

        article_tokens = []
        paragraph_tokens = []
        sentence_spans = []

        #TODO: Write this loop, etc in proper recursive form.

        for i,article in enumerate(articles):

            print 'Processing article in', self.articles_meta[i]

            paragraphs = paragraph_tokenize(article)
            
            for paragraph in paragraphs:
                sentences = sentence_tokenize(paragraph)

                for sentence in sentences:
                    word_tokens = word_tokenize(sentence)

                    self.word_tokens.extend(word_tokens)
                    
                    sentence_spans.append(len(word_tokens))

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

    @property
    def tokens_metadata(self):

        d = dict(articles = self.articles_meta,
                 paragraphs = None,
                 sentences = None)

        return d


class IepTokens(SepTokens):
    pass







############################
####      TEST DATA    #####
############################

def genBaseCorpus():
    return BaseCorpus(['cats','chase','dogs','dogs','do','not','chase','cats'], {'sentences' : [3]})


def test_IepTokens():

    path = 'test-data/iep-selected'

    tokens = IepTokens(path)

    print 'Article breaks:\n', tokens.articles
    print '\nParagraph breaks:\n', tokens.paragraphs
    print '\nSentence breaks:\n', tokens.sentences

    return tokens


def test_BaseCorpus():

    path = 'test-data/iep-selected'

    tokens = IepTokens(path)

    c = BaseCorpus(tokens.word_tokens, tokens.tokens_dict)

    print 'First article:\n', c.view_tokens('articles')[1]
    print '\nFirst five paragraphs:\n', c.view_tokens('paragraphs')[:5]
    print '\nFirst ten sentences:\n', c.view_tokens('sentences')[:10]

    print '\nLast article:\n', c.view_tokens('articles')[-1]
    print '\nLast five paragraphs:\n', c.view_tokens('paragraphs')[-5:]
    print '\nLast ten sentences:\n', c.view_tokens('sentences')[-10:]

    return c


def test_integer_corpus():

    path = 'test-data/iep-selected'

    tokens = IepTokens(path)

    c = BaseCorpus(tokens.word_tokens, tokens.tokens_dict)

    int_corpus, decoder = c.digitize()

    print 'First article:\n',\
          int_corpus.view_tokens('articles', decoder)[1]
    print '\nFirst five paragraphs:\n',\
          int_corpus.view_tokens('paragraphs', decoder)[:5]
    print '\nFirst ten sentences:\n',\
          int_corpus.view_tokens('sentences', decoder)[:10]

    print '\nLast article:\n',\
          int_corpus.view_tokens('articles', decoder)[-1]
    print '\nLast five paragraphs:\n',\
          int_corpus.view_tokens('paragraphs', decoder)[-5:]
    print '\nLast ten sentences:\n',\
          int_corpus.view_tokens('sentences', decoder)[-10:]

    return int_corpus, decoder


def test_Corpus():

    path = 'test-data/iep-selected'

    tokens = IepTokens(path)

    c = Corpus(tokens.word_tokens, tokens.tokens_dict,
               tokens.tokens_metadata)

    print 'First article:\n',\
          c.view_tokens('articles', True)[1]
    print '\nFirst five paragraphs:\n',\
          c.view_tokens('paragraphs', True)[:5]
    print '\nFirst ten sentences:\n',\
          c.view_tokens('sentences', True)[:10]

    print '\nLast article:\n',\
          c.view_tokens('articles', True)[-1]
    print '\nLast five paragraphs:\n',\
          c.view_tokens('paragraphs', True)[-5:]
    print '\nLast ten sentences:\n',\
          c.view_tokens('sentences', True)[-10:]

    print '\nSource of second article:',\
          c.view_metadata('articles')[2]

    return c
