from inphosemantics import corpus
from inphosemantics.corpus import tokenizer

from couchdb import * # experiment database
from couchdb.mapping import * # Object Relational Mapper


from datetime import datetime


## iep_test_plain_path
#plain_path = 'inphosemantics/tests/data/iep/selected/corpus/plain'

#stoplist_path = 'inphosemantics/tests/data/stoplists/stoplist-nltk-english.txt'



## experiment boilerplate
couch = Server()
database = couch['inphosemantics']

plain_path = database['iep_test']['plain_path']
stoplist_file = database['iep_test']['stoplists']['nltk']


############
## Utilities
############
tokens = tokenizer.ArticlesTokenizer(plain_path)
terms = tokens.terms
tok_names = tokens.tok_names
tok_data = tokens.tok_data





def load_test_stoplist():

    with open(stoplist_path, 'r') as f:
        
        stoplist = f.read().split('\n')

    stoplist = [word for word in stoplist if word]

    return stoplist


###################################
## CouchDB Document Wrapper Classes
###################################


## Used to represent a general entry in our Inpho CouchDB Database.
## Eventually the aim is to have this generalized out of tests_corpus.py
## and into somewhere more neutral to the project setup.
class Metadata(Document):
    added = DateTimeField(default=datetime.now)
    type  = TextField(default="metadata")
    

## Currently represents a Corpus mapping in the CouchDB database.
## Though set up as a CouchDB object-relational mapping,
## this object is not the actual object manipulated elsewhere.
## In that sense, it's not true ORM, but rather intended to serve as
## a pointer to where the Corpus data is archived on disk.
class CorpusMeta(Metadata):
    plain_path = TextField()
    raw_path   = TextField()
    stoplists  = ListField(TextField)
    compressed = BooleanField(default=False)
    masking_functions = ListField(TextField)
    short_label = TextField()
    long_label = TextField()
    type = TextField(default="Corpus")
    
    
###############
## Corpus Tests
###############
def test_corpus():

    c = None # So that c exists before the experiment.

    ## run the experiment
    try:
        c = corpus.Corpus(terms,
                   tok_names=tok_names,
                   tok_data=tok_data)

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

    ## Catch any problems
    except:
        raise

    ## Return the result
    return c






# MaskedCorpus tests


def test_masked_corpus_1():

    c = None # So that C exists before the test

    ## Run the test
    try:
        text = ['I', 'came', 'I', 'saw', 'I', 'conquered']
        tok_names = ['sentences']
        tok_data = [[(2, 'Veni'), (4, 'Vidi'), (6, 'Vici')]]
        masked_terms = ['I', 'came']
        
        c = corpus.MaskedCorpus(text,
                                tok_names=tok_names,
                                tok_data=tok_data,
                                masked_terms=masked_terms)
        
        from tempfile import TemporaryFile
        tmp = TemporaryFile()
        c.save(tmp)
        
        tmp.seek(0)
        
        c_in = corpus.MaskedCorpus.load(tmp)
        
        print 'corpus:', c_in.corpus
        print 'data:', c_in.corpus.data
        print 'mask:', c_in.corpus.mask
        print 'fill value:', c_in.corpus.fill_value
        print 'tokenizations:', c_in.tok
        print 'terms:', c_in.terms
        print 'terms str->int:', c_in.terms_int
        print 'masked terms:', c_in.masked_terms
        print 'compressed_corpus:', c_in.compressed_corpus()

    except:
        raise
    
    # Return the result
    return c_in




def test_masked_corpus_2():

    c = None # So that c exists before the test

    ## Run the test
    try:
        
        c = corpus.MaskedCorpus(terms,
                                tok_names=tok_names,
                                tok_data=tok_data)
        
        stoplist = load_test_stoplist()
        
        corpus.mask_sing_occ(c)
        
        corpus.mask_from_stoplist(c, stoplist)
        
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

    except:
        raise
    
    return c




def test_compressed():

    c = test_masked_corpus_2()

    comp_c = c.compressed_corpus()

    for i in xrange(10):

        print '\nFlat view:'
        print 'Token type', type(c.view_tokens('sentences', True)[i])
        print [str(ma.data) for ma in c.view_tokens('sentences', True)[i]]

        print '\nMasked view:'
        print c.view_tokens('sentences', True)[i]

        print '\nCompressed view:'
        print comp_c.view_tokens('sentences', True)[i]



def test_masked_corpus_save():

    c = None # So that c exists before running the test.
    
    ## Run the test.
    try:
        doc = CorpusMeta(plain_path=plain_path, compressed=False) ## CouchDB entry


        ## Use the package defaults for terms & tokens
        c = corpus.MaskedCorpus(terms,
                                tok_names=tok_names,
                                tok_data=tok_data)


        ## Fetch the stoplist and record it
        stoplist = load_test_stoplist()
        corpus.mask_from_stoplist(c, stoplist)
        
        ## Save the file
        corpus_filename = 'inphosemantics/tests/data/iep/selected/corpus/iep-selected-nltk.npz'
        c.save(corpus_filename)
        
        db.save(doc)
        
    except:
        raise

    return c


def test_masked_corpus_save_compressed():
    c = None # So that c exists before the test runs
    try:
        
        c = corpus.MaskedCorpus(terms,
                                tok_names=tok_names,
                                tok_data=tok_data)
        
        stoplist = load_test_stoplist()
        corpus.mask_from_stoplist(c, stoplist)
        
        corpus_filename_compressed = 'inphosemantics/tests/data/iep/selected/corpus/iep-selected-nltk-compressed.npz'
        c.save(corpus_filename_compressed, compressed=True)
        
    except:
        raise

    return c
