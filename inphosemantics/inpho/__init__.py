from couchdb import Document, Server
from couchdb.mapping import *
from inphosemantics import corpus
from inphosemantics.model import tf

__all__ = ['FileDocument', 'CorpusDocument', 'MatrixDocument']



server = Server()

inpho_db = server['inpho']


class FileDocument(Document):

    date = DateTimeField(default=datetime.now)

    filename = TextField()



class MaskedCorpusDocument(FileDocument):

    name = TextField()

    masking_fns = ListField(TextField())

    src_plain_dir = TextField()


class CompressedCorpusDocument(FileDocument):

    name = TextField()

    src_maskedcorp_file = TextField()



class MatrixDocument(FileDocument):

    name = TextField()

    filename = TextField()
    
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

    return doc




def tf_trainer(corpus_name, tok_name, masking_fns=[]):

    ## Fetch a (CouchDB) view of qualified corpora
    corpus_view = inpho_db.query(
        '''
        function(doc){
          if (doc.name === '%s'){
            emit(doc.name, doc)
          }
        }
        ''' % corpus_name
    )

    ## Get the corpus meta data and fetch the corpus from disk.
    corpus_doc = corpus_view.rows[0].value
    sep_corpus = corpus.Corpus.load(corpus_doc['filename'])

    ## Fetch the model and train it.
    tfModel = tf.TfModel()
    tfModel.train(sep_corpus, tok_name)
    
    ## Save the matrix to disk and record
    ## its existence in the database.

    #matrix_dir = inpho_db['data_root']['dir'] + corpus_name + '/matrices/'
    matrix_dir = inpho_db['data_root']['dir'] + 'sep/complete/matrices/'
    
    matrix_filename = corpus_name + '-'
    for fn in masking_fns:
        matrix_filename = matrix_filename + fn + '-'
    matrix_filename = matrix_filename + 'tf-' + tok_name + '.npy'

    matrix_path = matrix_dir + matrix_filename
    
    tfModel.save_matrix(matrix_path)

    matrix_doc = MatrixDocument(name='tf',
                                model_class=TfModel.__name__,
                                filename=matrix_path,
                                src_corpus_file=corpus_doc['filename'],
                                tok_name=tok_name)
    
    inpho_db.save(matrix_doc)

    return tfModel


# Typical call:
# >>> tf_trainer('sep', masking_fns=['nltk'], tok_name='articles')


# Typical call:
# >>> tf_viewer('sep', tok_name='paragraphs', masking_fns=['nltk'])

def tf_viewer(corpus_name, tok_name, masking_fns=[]):
    ## valid tok_name values include 'paragraphs', 'articles', ...

    ## Fetch a (CouchDB) view of qualified corpora.
    corpus_view = inpho_db.query(
        '''
        function(doc){
          if (doc.name === '%s'){
            emit(doc.name, doc);
          }
        }
        ''' % corpus_name
    )

    ## Get the corpus meta data and fetch the corpus from disk.
    corpus_doc = corpus_view.rows[0].value
    corpus_obj = corpus.Corpus.load(corpus_doc['filename'])



    ## Fetch a (CouchDB) view of qualified matrices.
    matrix_view = inpho_db.query(
        '''
        function(doc){
          if (doc.name === 'tf' && tok_name === '%s' ){
            emit(doc.name, doc);
          }
        }
        ''' % tok_name
    )

    ## Get the matrix meta data and fetch the matrix from disk.
    matrix_doc = matrix_view.rows[0].value
    matrix_path = matrix_doc['filename']

    ## fetch the matrix given
    viewer = TfViewer(tok_name=tok_name)

    ## Now that we have the filename of the matrix, load it
    viewer.load_matrix(matrix_path)
    
    return viewer


def InphoTrainer(object):

    pass


def InphoViewer(object):

    pass

