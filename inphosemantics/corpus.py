import os.path
import pickle
from nltk.corpus import stopwords as nltk_stopwords


class Corpus(object):

    data_root = '/var/inphosemantics/data/'

    def __init__(self, corpus, corpus_param):

        self.corpus = corpus
        self.corpus_param = corpus_param
    
        self.corpus_path =\
            os.path.join(Model.data_root, self.corpus, 
                         self.corpus_param, 'corpus')

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


    def tokenized_sentences(self, title):

        tokenized_path = os.path.join(self.corpus_path, 'tokenized')
        
        return

    
    def plain_text(self, title):

        plain_path = os.path.join(self.corpus_path, 'plain')

        return


    def raw(self, title):

        raw_path = os.path.join(self.corpus_path, 'html')

        return
