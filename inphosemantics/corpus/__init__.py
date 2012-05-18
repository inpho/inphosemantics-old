import bz2
import pickle

import numpy as np

from inphosemantics import *



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

        self.filename = filename
        

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

        f = bz2.BZ2File(self.filename, 'w')
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

        digitized_corpus =\
            BaseCorpus(int_corp, tokens_dict=self.tokens_dict,
                       dtype=np.uint32)
    
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
                    
                    if i < len(v)-1 and j > v[i+1]: 
                        raise Exception('malsorted tokenization for ' + str(k)
                                        + ': tokens ' + str(j)
                                        + ' and ' + str(v[i+1]))
                    
                    if j >= len(self.term_tokens):
                        print v[-30:]
                        raise Exception('invalid tokenization for ' + str(k)
                                        + ': ' + str(j) + ' is out of range ('
                                        + str(len(self.term_tokens)) + ')')

                    #TODO: Define a proper exception

        return True



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



############################
####      TEST DATA    #####
############################

def genBaseCorpus():
    return BaseCorpus(['cats','chase','dogs','dogs','do','not','chase','cats'], {'sentences' : [3]})



def test_BaseCorpus():

    from inphosemantics.corpus.tokenizer import IepTokens

    path = 'test-data/iep/selected/corpus/plain'

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

    from inphosemantics.corpus.tokenizer import IepTokens

    path = 'test-data/iep/selected/corpus/plain'

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

    from inphosemantics.corpus.tokenizer import IepTokens

    path = 'test-data/iep/selected/corpus/plain'

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


def test_Corpus_dump():

    from inphosemantics.corpus.tokenizer import IepTokens

    path = 'test-data/iep/selected/corpus/plain'
    filename = 'test-data/iep/selected/corpus/iep-selected.pickle.bz2'

    tokens = IepTokens(path)

    c = Corpus(tokens.word_tokens, tokens.tokens_dict,
               tokens.tokens_metadata, filename)

    c.dumpz()


# import codecs
# import os.path

# from nltk.corpus import stopwords as nltk_stopwords

# class CorpusBase(object):

#     data_root = '/var/inphosemantics/data/'

#     def __init__(self, corpus, corpus_param):
        
#         self.corpus = corpus
#         self.corpus_param = corpus_param
        
#         self.corpus_path =\
#             os.path.join(Corpus.data_root, self.corpus, 
#                          self.corpus_param, 'corpus')

#         self.tokenized_path = os.path.join(self.corpus_path, 'tokenized')
#         self.plain_path = os.path.join(self.corpus_path, 'plain')


#     #TODO: Collapse these three into one function (if indeed all three
#     #are needed after refactoring),
#     def tokenized_document(self, name):

#         if os.path.splitext(name)[1] != '.pickle':
#             name += '.pickle'

#         tokenized_file = os.path.join(self.tokenized_path, name)

#         print 'Reading tokenized document from', tokenized_file
#         with open(tokenized_file, 'r') as f:
#             return pickle.load(f)


#     def tokenized_sentences(self, name):

#         if os.path.splitext(name)[1] != '.pickle':
#             name += '.pickle'

#         tokenized_file = os.path.join(self.tokenized_path, name)

#         print 'Reading tokenized sentences from', tokenized_file
#         with open(tokenized_file, 'r') as f:
#             return [sent for para in pickle.load(f) for sent in para]


#     def tokenized_paragraphs(self, name):

#         if os.path.splitext(name)[1] != '.pickle':
#             name += '.pickle'

#         tokenized_file = os.path.join(self.tokenized_path, name)

#         print 'Reading tokenized paragraphs from', tokenized_file
#         with open(tokenized_file, 'r') as f:
            
#             document = pickle.load(f)

#             for i,paragraph in enumerate(document):
#                 document[i] = [word for sent in paragraph for word in sent]

#             return document

    
#     def plain_text(self, name):

#         if os.path.splitext(name)[1] != '.txt':
#             name = name + '.txt'

#         plain_file = os.path.join(self.plain_path, name)

#         print 'Loading plain text file for the article', '\'' + name + '\''
#         with codecs.open(plain_file, encoding='utf-8', mode='r') as f:
#             return f.read()


#     def raw(self, name):
#         """
#         'name' here denotes the name of the html subdirectory directly
#         containing the article
#         """

#         # TODO: For now, 'raw' is actually 'html'; this should be
#         # generalized.

#         raw_file =\
#             os.path.join(self.corpus_path, 'html', name, 'index.html')
#         print 'Loading HTML file for the article', '\'' + name + '\''

#         with codecs.open(raw_file, encoding='utf-8', mode='r') as f:
#             return f.read()



# class Corpus(CorpusBase):

#     def __init__(self, corpus, corpus_param):

#         CorpusBase.__init__(self, corpus, corpus_param)

#         self.lexicon = self.lexicon()
#         self.stopwords = self.stopwords()


#     def lexicon(self):

#         lexicon_filename =\
#             '-'.join([self.corpus, self.corpus_param, 'lexicon.pickle'])
#         lexicon_file =\
#             os.path.join(self.corpus_path, 'lexicon', lexicon_filename)

#         print 'Reading lexicon'
#         with open(lexicon_file, 'r') as f:
#             return pickle.load(f)


#     def stopwords(self):

#         # TODO: Manage stop lists properly.
#         beagle_stop_file =\
#             os.path.join(__path__[0], 'beagle-stopwords-jones.txt')

#         with open(beagle_stop_file, 'r') as f:
#             beagle_stop = f.read()
        
#         beagle_stop = beagle_stop.split()

#         ext_stop =\
#             ['especially', 'many', 'several', 'perhaps', 
#              'various', 'key', 'found', 'particularly', 'later', 
#              'could', 'might', 'must', 'would', 'may', 'actually',
#              'either', 'without', 'one', 'also', 'neither', 'well',
#              'including', 'although', 'much', 'largely', 'clearly', 'thus',
#              'since', 'regarded', 'indeed', 'however', 'rather',
#              'ultimately', 'yet', 'according', 'nevertheless', 'finally',
#              'concerning', 'cf', 'seen', 'primarily', 'conversely',
#              'relatedly', 'subsequent']

#         print 'Reading stop words'

#         stopwords = set(beagle_stop + ext_stop
#                         + nltk_stopwords.words('english'))
#         lexicon = set(self.lexicon)

#         return list(stopwords.intersection(lexicon))
