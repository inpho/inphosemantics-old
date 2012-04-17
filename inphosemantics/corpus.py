import os.path
import pickle
import codecs
from nltk.corpus import stopwords as nltk_stopwords


class CorpusBase(object):

    data_root = '/var/inphosemantics/data/'

    def __init__(self, corpus, corpus_param):
        
        self.corpus = corpus
        self.corpus_param = corpus_param
        
        self.corpus_path =\
            os.path.join(Corpus.data_root, self.corpus, 
                         self.corpus_param, 'corpus')

        self.tokenized_path = os.path.join(self.corpus_path, 'tokenized')
        self.plain_path = os.path.join(self.corpus_path, 'plain')


    def tokenized_sentences(self, name):

        if os.path.splitext(name)[1] != '.pickle':
            name += '.pickle'

        tokenized_file = os.path.join(tokenized_path, name)

        print 'Reading tokenized sentences from', tokenized_file
        with open(tokenized_file, 'r') as f:
            return [sent for para in pickle.load(f) for sent in para]

    
    def plain_text(self, name):

        if os.path.splitext(name)[1] != '.txt':
            name = name + '.txt'

        plain_file = os.path.join(plain_path, name)

        print 'Loading plain text file for the article', '\'' + name + '\''
        with codecs.open(plain_file, encoding='utf-8', mode='r') as f:
            return f.read()


    def raw(self, name):
        """
        'name' here denotes the name of the html subdirectory directly
        containing the article
        """

        # TODO: For now, 'raw' is actually 'html'; this should be
        # generalized.

        raw_file =\
            os.path.join(self.corpus_path, 'html', name, 'index.html')
        print 'Loading HTML file for the article', '\'' + name + '\''

        with codecs.open(raw_file, encoding='utf-8', mode='r') as f:
            return f.read()



class Corpus(CorpusBase):

    def __init__(self, corpus, corpus_param):

        CorpusBase.__init__(self, corpus, corpus_param)

        self.lexicon = self.lexicon()
        self.stopwords = self.stopwords()


    def lexicon(self):

        lexicon_filename =\
            '-'.join([self.corpus, self.corpus_param, 'lexicon.pickle'])
        lexicon_file =\
            os.path.join(self.corpus_path, 'lexicon', lexicon_filename)

        print 'Reading lexicon'
        with open(lexicon_file, 'r') as f:
            return pickle.load(f)


    def stopwords(self):

        # TODO: Manage stop lists properly.
        ext_stop =\
            ['especially', 'many', 'several', 'perhaps', 
             'various', 'key', 'found', 'particularly', 'later', 
             'could', 'might', 'must', 'would', 'may', 'actually',
             'either', 'without', 'one', 'also', 'neither']

        print 'Reading stop words'
        return ext_stop + nltk_stopwords.words('english')
