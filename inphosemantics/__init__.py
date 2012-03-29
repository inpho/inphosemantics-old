import os
import pickle
import codecs
import tempfile
from multiprocessing import Pool

import numpy as np
from math import sqrt

from nltk.corpus import stopwords as nltkstopwords
from xml.etree.ElementTree import ElementTree 


######################################################################

data = '/var/inphosemantics/data'

def change_root(path, newroot):
    components = []
    while path and path != '/':
        components.insert(0, os.path.basename(path))
        path = os.path.dirname(path)
    return os.path.join(newroot, *components[1:])


def gen_filename(modpath, base):
    # Expects the arguments to be the value of __path__ for a
    # submodule and a standard name for the type of file with the
    # extension e.g., 'vector.pickle' or 'lexicon.pickle'
    change_root(modpath, '')
    components = []
    while modpath and modpath != '/':
        components.insert(0, os.path.basename(modpath))
        modpath = os.path.dirname(modpath)
    return '-'.join(components + [base])


######################################################################


def mk_read_lexicon(path):
    
    datapath = os.path.join(change_root(path, data), 'lexicon')
    filename = gen_filename(change_root(path, ''), 'lexicon.pickle')

    def read_lexicon():
        print 'Reading lexicon'
        path = os.path.join(datapath, filename)
        with open(path, 'r') as f:
            return pickle.load(f)

    return read_lexicon


def mk_write_lexicon(path):
    
    datapath = os.path.join(change_root(path, data), 'lexicon')
    filename = gen_filename(change_root(path, ''), 'lexicon.pickle')

    def write_lexicon(lexicon):
        path = os.path.join(datapath, filename)
        print 'Writing lexicon to', path
        with open(path, 'w') as f:
            pickle.dump(lexicon, f)

    return write_lexicon


def get_tokpath(path):
    return os.path.join(change_root(path, data), 'tokenized')


def mk_read_sentences(path):

    def read_sentences(filename):
        if os.path.splitext(filename)[1] != '.pickle':
            filename += '.pickle'
        filename = os.path.join(get_tokpath(path), filename)
        print 'Reading tokenized sentences from', filename
        with open(filename, 'r') as f:
            return [sent for para in pickle.load(f) for sent in para]

    return read_sentences


def mk_write_tokens(path):

    def write_tokens(sents, title):

        filename = os.path.join(get_tokpath(path), title + '.pickle')

        print 'Writing tokenized paragraphs and sentences to', filename
        with open(filename, 'w') as f:
            pickle.dump(sents, f)

    return write_tokens


def get_htmlpath(path):
    return os.path.join(change_root(path, data), 'html')


def mk_read_html(path):

    def read_html(title):
        """
        'title' here denotes the name of the directory directly
        containing the article
        """
        filename = os.path.join(get_htmlpath(path), title, 'index.html')
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

    return read_html


def get_plainpath(path):
    return os.path.join(change_root(path, data), 'plain')


def mk_read_plain(path):

    def read_plain(title):

        if os.path.splitext(title)[1] != '.txt':
            filename = title + '.txt'
        else:
            filename = title

        filename = os.path.join(get_plainpath(path), filename)

        print 'Loading plain text file for the article', '\'' + title + '\''
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            return f.read()

    return read_plain


def mk_write_plain(path):

    def write_plain(article, title):

        if os.path.splitext(title) != '.txt':
            filename = title + '.txt'
        else:
            filename = title

        filename = os.path.join(get_plainpath(path), filename)

        print 'Writing plain text file for the article', '\'' + title + '\''
        with codecs.open(filename, encoding='utf-8', mode='w') as f:
            f.write(article)
        return

    return write_plain




def mk_read_stopwords():

    def read_stopwords():
        print 'Reading stop words'
        return nltkstopwords.words('english')

    return read_stopwords


# stopfile = 'corpus/stopwords/stopwords.txt'

# def mk_read_stopwords(path):
#     def read_stopwords():
#         print 'Reading stop words'
#         with codecs.open(path, encoding='utf-8', mode='r') as f:
#             corpstop = set(f.read(f).split())
#             nltkstop = set(nltkstopwords.words('english'))
#             return corpstop.union(nltkstop)
#     return read_stopwords

######################################################################

def mk_read_vecs(path):

    datapath = change_root(path, data)
    filename = gen_filename(change_root(path, ''), 'vectors.pickle')

    def read_vecs():
        print 'Reading vectors'
        path = os.path.join(datapath, filename)
        with open(path, 'r') as f:
            return pickle.load(f)

    return read_vecs

def mk_write_vecs(path):

    datapath = change_root(path, data)
    filename = gen_filename(change_root(path, ''), 'vectors.pickle')

    def write_vecs(vecs):
        print 'Writing vectors'
        path = os.path.join(datapath, filename)
        with open(path, 'w') as f:
            pickle.dump(vecs, f)
            return

    return write_vecs

######################################################################

def norm(v):
    return sqrt(np.dot(v,v))

def normalize(v):
    if np.any(v):
        return v / norm(v) # component-wise division
    else:
        return v # would be division by zero

def vector_cos(v,w):
    '''
    Computes the cosine of the angle between vectors v and w. A result
    of -2 denotes undefined, i.e., when v or w is a zero vector
    '''
    if not v.any() or not w.any():
        return -2
    else:
        return np.dot(v,w) / (norm(v) * norm(w))

######################################################################


def mk_read_simvec(path):

    datapath = os.path.join(change_root(path, data), 'cosines')

    def read_simvec(index):
        name = gen_filename(
            change_root(path, ''), 'cosines-' + str(index) + '.pickle')
        name = os.path.join(datapath, name)
        print 'Reading similarity vector', name
        with open(name, mode='r') as f:
            return pickle.load(f)

    return read_simvec


def mk_write_simvec(path):

    datapath = os.path.join(change_root(path, data), 'cosines')

    def write_simvec(vector, index):
        name = gen_filename(
            change_root(path, ''), 'cosines-' + str(index) + '.pickle')
        name = os.path.join(datapath, name)
        print 'Writing similarity vector', name
        with open(name, mode='w') as f:
            pickle.dump(vector, f)
            return

    return write_simvec


def init_gen_cosine(path):

    vecs = mk_read_vecs(path)()
    vecs = [np.float32(v) for v in vecs]
    vecdict = dict(zip([tuple(v) for v in vecs], xrange(len(vecs))))
    gen_cosine.write_simvec = mk_write_simvec(path)
    gen_cosine.vecs = vecs
    gen_cosine.vecdict = vecdict
    return


def gen_cosine(v1):
    v3 = map(lambda v2: vector_cos(v1, v2), gen_cosine.vecs)
    v3 = np.array(v3, dtype=np.float32)
    index = gen_cosine.vecdict[tuple(v1)]
    gen_cosine.write_simvec(v3, index)
    return


def mk_gen_cosines(path):

    def gen_cosines(n = -1):

        init_gen_cosine(path)
        vecs = gen_cosine.vecs

        if n == -1:
            n = len(vecs)

        p = Pool()
        p.map(gen_cosine, vecs[:n], 100)
        p.close()

        return

    return gen_cosines
        
######################################################################
