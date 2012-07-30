from inphosemantics import corpus
from inphosemantics.corpus import tokenizer

from couchdb import * # experiment database
from couchdb.mapping import * # Object Relational Mapper


from datetime import datetime


## iep_test_plain_path
#plain_path = 'inphosemantics/tests/data/iep/selected/corpus/plain'

#stoplist_path = 'inphosemantics/tests/data/stoplists/stoplist-nltk-english.txt'

corpus_filename = 'inphosemantics/tests/data/iep/selected/corpus/'\
                  'iep-selected-nltk.npz'

corpus_filename_compressed = 'inphosemantics/tests/data/iep/selected/corpus/'\
                             'iep-selected-nltk-compressed.npz'



## experiment boilerplate
couchServer = Server()
db = couchServer['inphosemantics']



## TODO: Set up this data to be fetched
## Fetch some test data
plain_path = db['iep_test']['plain_path']

stoplist_file = db['iep_test']['stoplists']['nltk']

#corpus_filename = db['iep_test'][]


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
    type  = TextField()
    

## Currently represents a Corpus mapping in the CouchDB database.
## Though set up as a CouchDB object-relational mapping,
## this object is not the actual object manipulated elsewhere.
## In that sense, it's not true ORM, but rather intended to serve as
## a pointer to where the Corpus data is archived on disk.
class CorpusMeta(Metadata):
    plain_path = TextField()
    raw_path   = TextField()
    stoplists  = ListField(TextField)
    isCompressed = BooleanField(default=False)
    masking_functions = ListField(TextField)
    short_label = TextField()
    long_label = TextField()

    
    
###############
## Corpus Tests
###############
def test_corpus():

    ## run the experiment
    try:
        terms, tok_names, tok_data = tokenize_test_corpus()
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
        pass

    ## Return the result
    return c






# MaskedCorpus tests


def test_masked_corpus_1():
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
    
    # Return the result
    return c_in






def test_masked_corpus_2():

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
    c = corpus.MaskedCorpus(terms,
                            tok_names=tok_names,
                            tok_data=tok_data)
    
    stoplist = load_test_stoplist()
    
    corpus.mask_from_stoplist(c, stoplist)
    
    c.save(corpus_filename)


def test_masked_corpus_save_compressed():
    c = corpus.MaskedCorpus(terms,
                     tok_names=tok_names,
                     tok_data=tok_data)

    stoplist = load_test_stoplist()

    corpus.mask_from_stoplist(c, stoplist)

    c.save(corpus_filename_compressed, compressed=True)

