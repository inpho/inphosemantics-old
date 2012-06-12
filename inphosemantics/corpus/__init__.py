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
                    
                    if j >= len(self.corpus):
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
    Corpus is an instance of BaseCorpus with datatype int32 that
    bundles various useful data and methods associated with a given
    instance of BaseCorpus.

    metadata associated with any given tokenization: terms, sentences,
    etc.

    Expected usage. corpus would be the atomic tokens (e.g.,
    words) as strings. Instantiation of Corpus will map the
    corpus to integer representations of the terms.

    terms is the indexed set of strings occurring in corpus. It is
    string-typed numpy array.

    terms_int is a mapping object whose keys are given by terms and
    whose values are their corresponding integers
    
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





    def view_metadata(self, name):

        return self.tokens_meta[name]




    def gen_lexicon(self):
        """
        Create a corpus object that contains the term types alone (as
        integers and as strings)
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


def test_BaseCorpus():

    from inphosemantics.corpus.tokenizer import IepTokens

    path = 'test-data/iep/selected/corpus/plain'

    tokens = IepTokens(path)

    c = BaseCorpus(tokens.word_tokens, tokens.tokens)

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

    c = BaseCorpus(tokens.word_tokens, tokens.tokens)

    int_corpus, encoder = c.encode_corpus()

    print 'First article:\n',\
          int_corpus.view_tokens('articles', encoder)[1]
    print '\nFirst five paragraphs:\n',\
          int_corpus.view_tokens('paragraphs', encoder)[:5]
    print '\nFirst ten sentences:\n',\
          int_corpus.view_tokens('sentences', encoder)[:10]

    print '\nLast article:\n',\
          int_corpus.view_tokens('articles', encoder)[-1]
    print '\nLast five paragraphs:\n',\
          int_corpus.view_tokens('paragraphs', encoder)[-5:]
    print '\nLast ten sentences:\n',\
          int_corpus.view_tokens('sentences', encoder)[-10:]

    return int_corpus, encoder


def test_Corpus():

    from inphosemantics.corpus.tokenizer import IepTokens

    path = 'test-data/iep/selected/corpus/plain'

    tokens = IepTokens(path)

    c = Corpus(tokens.word_tokens, tokens.tokens,
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


def test_Corpus_dumpz():

    from inphosemantics.corpus.tokenizer import IepTokens

    path = 'test-data/iep/selected/corpus/plain'
    filename = 'test-data/iep/selected/corpus/iep-selected.pickle.bz2'

    tokens = IepTokens(path)

    c = Corpus(tokens.word_tokens, tokens.tokens,
               tokens.tokens_metadata)

    c.dumpz(filename)


def test_Corpus_dumpz_plato():

    from inphosemantics.corpus.tokenizer import IepTokens

    path = 'test-data/iep/plato/corpus/plain'
    filename = 'test-data/iep/plato/corpus/iep-plato.pickle.bz2'

    tokens = IepTokens(path)

    c = Corpus(tokens.word_tokens, tokens.tokens,
               tokens.tokens_metadata)

    c.dumpz(filename)



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
