import re

from inphosemantics import corpus
from inphosemantics.corpus import tokenizer

from inphosemantics import model
from inphosemantics.model import tf
from inphosemantics.model import tfidf
from inphosemantics.model import lsa

from inphosemantics.viewer import tfviewer
from inphosemantics.viewer import tfidfviewer
from inphosemantics.viewer import lsaviewer



root = '/var/inphosemantics/data/'

test_root = 'inphosemantics/tests/data/'



class Inphodata(object):
    """
    """
    def __init__(self, root, corpus_name):



        corpus_subdir = re.sub(r'-', '/', corpus_name) + '/'
     
        corpus_path = root + corpus_subdir + 'corpus/'

        matrix_path = root + corpus_subdir + 'matrices/'



        self.plain_path = corpus_path + 'plain/'
                
        self.stoplist_nltk_filename = root + 'stoplists/'\
                                      'stoplist-nltk-english.txt'
        
        self.stoplist_jones_filename = root + 'stoplists/'\
                                       'stoplist-jones-beagle.txt'
        
        self.corpus_nltk_filename = corpus_path + corpus_name + '-nltk-f1.npz'
        
        self.compressed_corpus_nltk_filename = corpus_path + corpus_name +\
                                               '-nltk-f1-compressed.npz'
        
        self.corpus_jones_filename = corpus_path + corpus_name +\
                                     '-jones-f1.npz'
        
        self.compressed_corpus_jones_filename = corpus_path + corpus_name +\
                                                '-jones-f1-compressed.npz'
        
        self.tf_tok_name = 'paragraphs'
        
        self.tf_filename = matrix_path + corpus_name +\
                           '-nltk-tf-paragraphs.npy'
        
        self.tfidf_filename = matrix_path + corpus_name +\
                              '-nltk-tfidf-paragraphs.npy'

        self.lsa_filename = matrix_path + corpus_name +\
                            '-nltk-lsa-paragraphs-ev300.npy'
                
        self.beagle_env_filename = matrix_path + corpus_name +\
                                   '-nltk-beagle-env-d2048-l7.npy'
        
        self.beagle_ctx_filename = matrix_path + corpus_name +\
                                   '-nltk-beagle-ctx-d2048-l7.npy'
        
        self.beagle_ord_filename = matrix_path + corpus_name +\
                                   '-nltk-beagle-ord-d2048-l7.npy'

        self.beagle_comp_filename = matrix_path + corpus_name +\
                                    '-nltk-beagle-comp-d2048-l7.npy'


        
                      ### Corpus Processing ###

    def _proc_corpus(self,
                     stoplist_filename,
                     corpus_filename,
                     compressed_corpus_filename):
        
        # Tokenization

        tokens = tokenizer.ArticlesTokenizer(self.plain_path)

        c = corpus.MaskedCorpus(tokens.words,
                                tok_names=tokens.tok_names,
                                tok_data=tokens.tok_data)

        # Stoplist to mask

        with open(stoplist_filename, 'r') as f:
        
            stoplist = f.read().split('\n')

        stoplist = [word for word in stoplist if word]

        corpus.mask_from_stoplist(c, stoplist)

        corpus.mask_f1(c)

        # Write corpus files

        c.save(corpus_filename, compressed=False)

        c.save(compressed_corpus_filename, compressed=True)




    def proc_corpus_nltk(self):

        self._proc_corpus(self.stoplist_nltk_filename,
                          self.corpus_nltk_filename,
                          self.compressed_corpus_nltk_filename)


    def proc_corpus_jones(self):

        self._proc_corpus(self.stoplist_jones_filename,
                          self.corpus_jones_filename,
                          self.compressed_corpus_jones_filename)


    def get_corpus(self, stoplist_arg, compressed=False):
        """
        """
        if stoplist_arg == 'nltk':

            if compressed:

                return corpus.Corpus.load(self.compressed_corpus_nltk_filename)

            else:

                return corpus.MaskedCorpus.load(self.corpus_nltk_filename)

        if stoplist_arg == 'jones':

            if compressed:

                return corpus.Corpus.load(self.compressed_corpus_jones_filename)

            else:

                return corpus.MaskedCorpus.load(self.corpus_jones_filename)


        
                       ###  Term Frequency  ###
    
    def train_tf(self):

        c = corpus.Corpus.load(self.compressed_corpus_nltk_filename)

        m = tf.TfModel()

        m.train(c, self.tf_tok_name)

        m.save_matrix(self.tf_filename)



    def tf_viewer(self):

        v = tfviewer.TfViewer()

        v.load_corpus(self.compressed_corpus_nltk_filename)

        v.load_matrix(self.tf_filename)

        v.tok_name = self.tf_tok_name

        return v



                           ###  TF-IDF  ###

    def train_tfidf(self):

        c = corpus.Corpus.load(self.compressed_corpus_nltk_filename)

        m = tfidf.TfIdfModel()

        tf_matrix = model.Model.load_matrix(self.tf_filename)

        m.train(c, self.tf_tok_name, tf_matrix=tf_matrix)

        m.save_matrix(self.tfidf_filename)


        
    def tfidf_viewer(self):

        v = tfidfviewer.TfIdfViewer()
        
        v.load_corpus(self.compressed_corpus_nltk_filename)
        
        v.load_matrix(self.tfidf_filename)
        
        v.tok_name = self.tf_tok_name

        return v



                           ###  LSA  ###

    def train_lsa(self):

        c = corpus.Corpus.load(self.compressed_corpus_nltk_filename)

        m = lsa.LsaModel()

        tfidf_matrix = model.Model.load_matrix(self.tfidf_filename)

        m.train(c, self.tf_tok_name, tfidf_matrix=tfidf_matrix)

        m.save_matrix(self.lsa_filename)


        
    def lsa_viewer(self):

        v = lsaviewer.LsaViewer()
        
        v.load_corpus(self.compressed_corpus_nltk_filename)
        
        v.load_matrix(self.lsa_filename)
        
        v.tok_name = self.tf_tok_name

        return v

