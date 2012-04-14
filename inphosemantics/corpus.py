import os.path
import pickle
import tempfile
import codecs
from xml.etree.ElementTree import ElementTree
from nltk.corpus import stopwords as nltk_stopwords


class Corpus(object):

    data_root = '/var/inphosemantics/data/'

    def __init__(self, corpus, corpus_param):

        self.corpus = corpus
        self.corpus_param = corpus_param
    
        self.corpus_path =\
            os.path.join(Corpus.data_root, self.corpus, 
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


    def tokenized_sentences(self, name):

        tok_path = os.path.join(self.corpus_path, 'tokenized')

        if os.path.splitext(name)[1] != '.pickle':
            name += '.pickle'

        tok_file = os.path.join(tok_path, name)

        print 'Reading tokenized sentences from', tok_file
        with open(tok_file, 'r') as f:
            return [sent for para in pickle.load(f) for sent in para]

    
    def plain_text(self, name):

        plain_path = os.path.join(self.corpus_path, 'plain')

        if os.path.splitext(name)[1] != '.txt':
            name = name + '.txt'

        plain_file = os.path.join(plain_path, name)

        print 'Loading plain text file for the article', '\'' + name + '\''
        with codecs.open(plain_file, encoding='utf-8', mode='r') as f:
            return f.read()


    def raw(self, name):

        raw_path = os.path.join(self.corpus_path, 'html')

        print 'To be written.'

        return


    def import_html(self, title):
        """
        'title' here denotes the name of the directory directly
        containing the article
        """

        filename =\
            os.path.join(self.corpus_path, 'html', title, 'index.html')
        print 'Loading HTML file for the article', '\'' + title + '\''
        
        #TODO Error-handling
        tmp = tempfile.NamedTemporaryFile()
        tidy = ' '.join(['tidy', '-qn', '-asxml', '--clean yes',
                         '--ascii-chars yes', '--char-encoding utf8'])
        command = '%s %s>%s 2>/dev/null' % (tidy, filename, tmp.name)
        os.system(command)
        tree = ElementTree()
        tree.parse(tmp.name)
        tmp.close()

        return tree
