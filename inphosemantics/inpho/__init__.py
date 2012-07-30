from couchdb import Document, Server
from couchdb.mapping import *

__all__ = ['FileDocument', 'CorpusDocument', 'MatrixDocument']



server = Server()

inpho_db = server['inpho']


class FileDocument(Document):

    date = DateTimeField()

    filename = TextField()



class MaskedCorpusDocument(FileDocument):

    name = TextField()

    # masking_fns = ListField(TextField())

    src_plain_dir = TextField()


class CompressedCorpusDocument(FileDocument):

    name = TextField()

    src_maskedcorp_file = TextField()



class MatrixDocument(FileDocument):

    name = TextField()

    model_class = TextField()

    src_corpus_file = TextField()


    ####################################
    #    TF-related parameters
    
    # this could also be called `column-label` 
    tok_name = TextField()


    ####################################
    #    TFIDF-related parameters
    src_tf_file = TextField()
    
    
    ####################################
    #    LSA-related parameters
    src_td_file = TextField()
    reduction_fn = TextField()


    ####################################
    #    LDA-related parameters


    ####################################
    #    BEAGLE-related parameters
    n_columns = IntegerField()


    #    BEAGLE Context, Order, Composite
    src_env_file = TextField()


    #    BEAGLE Order
    hol_width = IntegerField()


    #    BEAGLE Composite
    src_ctx_file = TextField()
    src_ord_file = TextField()
    


######################################################################

# def train_tf():

#     c = corpus.Corpus.load(compressed_corpus_filename)

#     m = tf.TfModel()

#     m.train(c, tf_tok_name)

#     m.save_matrix(tf_filename)


# root = '/var/inphosemantics/data/sep/complete/'



def test_add_comp_corpus():

    filename = '/var/inphosemantics/data/sep/complete/'\
               'corpus/sep-complete-nltk-compressed.npz'

    name = 'sep'

    src_maskedcorp_file = '/var/inphosemantics/data/sep/complete/'\
                          'corpus/sep-complete-nltk.npz'

    doc = CompressedCorpusDocument(filename=filename,
                                   name=name,
                                   src_maskedcorp_file=src_maskedcorp_file,
                                   nltk=True
                                   )
    
    doc.store(inpho_db)




def tf_trainer(corpus_name, masking_fns=[], tok_name='paragraphs'):

    corpus_filename = inpho_db.query(
        '''
        function(doc){
          if (doc.name === 'sep' && doc.nltk){
            emit(doc.name, doc)
          }
        }
        '''
    )

    print 'Corpus filename', corpus_filename



# Typical call:
# >>> tf_trainer('sep', masking_fns=['nltk'], tok_name='articles')



def InphoTrainer(object):

    pass


def InphoViewer(object):

    pass

