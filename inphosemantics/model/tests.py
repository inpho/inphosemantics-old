import numpy as np

from inphosemantics.model.intcorp import IntegerCorpus
from inphosemantics.model.tf import TFModel, TFIDFModel



def test_tf():

    from inphosemantics.model.intcorp import IntegerCorpus

    words = np.arange(28)
    docs = [2,3,5,7,11,13,17,19,23]
    c = IntegerCorpus(words, partitions={'documents': docs})
    print 'Words', c
    print 'Docs', c.view_partition('documents')
    m = TFModel(c)
    print 'Empty TD Matrix', m.td_matrix.__repr__()
    m.train()
    print 'Trained TF Model', m.td_matrix.todense()
    print 'Term similarity to 25', m.similar_terms(25)
    print 'Document similarity to 9', m.similar_documents(9)

    return m


        
def test_tfidf():

    from inphosemantics.model.intcorp import IntegerCorpus

    words = np.arange(28)
    docs = [2,3,5,7,11,13,17,19,23]
    c = IntegerCorpus(words, partitions={'documents': docs})
    print 'Words', c
    print 'Docs', c.view_partition('documents')
    m = TFIDFModel(c)
    print 'Empty TD Matrix', m.td_matrix.__repr__()
    m.train()
    print 'Trained TFIDF Model', m.td_matrix.todense()
    print 'Term similarity to 25', m.similar_terms(25)
    print 'Document similarity to 9', m.similar_documents(9)

    return m
