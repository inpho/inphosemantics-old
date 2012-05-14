import os.path
import sys
import pickle
from multiprocessing import Pool
import numpy as np
from numpy.dual import fft, ifft

from scipy.sparse import lil_matrix, issparse
from numpy import ndarray

from inphosemantics.model import ModelBase
from inphosemantics.corpus import Corpus
from inphosemantics.corpus.tokenizer import Tokenizer
from inphosemantics.localmath import vector_cos, normalize





class SparseModel(ModelBase):

    def __init__(self, corpus, corpus_param, model, model_param):
        
        self.corpus = corpus
        self.corpus_param = corpus_param
        self.model = model
        self.model_param = model_param
        
        self.matrix_filename = ''

        self.matrix_path = os.path.join(corpus, corpus_param, model, model_param, matrix_filename)


class DenseModel(ModelBase):

    pass





class Matrix(object):

    def __init__(self, num_rows, num_cols, filename):

        self.filename = filename
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.set_matrix()        


    def get_matrix(self):
        
        if self._matrix ==  None:
            self.load()
            
        return self._matrix

    def set_matrix(self, mat=None):
        
        self._matrix = mat

    matrix = property(get_matrix, set_matrix)



    def load(self):

        with open(self.filename, 'r') as f:
            self.matrix = pickle.load(f)


    def dump(self):

        with open(self.filename, 'w') as f:

            if issparse(self.matrix):
                pickle.dump(self.matrix, f)
            else:
                self.matrix.dump(f)

                
    def initialize(self, sparse=False, dtype=float):
        
        if sparse:
            self.matrix =\
                lil_matrix((self.num_rows, self.num_cols), dtype=dtype)
        else:
            self.matrix =\
                np.zeros((self.num_rows, self.num_cols), dtype=dtype)
                

    def get_row(self, i):

        if issparse(self.matrix):
            return self.matrix.getrow(i).toarray()
        else:
            return self.matrix[i]


    def update_row(self, i, v):
        
        if issparse(self.matrix):
            for j in v.nonzero():
                self.matrix[i,j] = v[j]
        else:
            self.matrix[i] = v



    def get_col(self, j):

        if issparse(self.matrix):
            return self.matrix.getcol(j).toarray()
        else:
            return self.matrix[:,j]


    def update_col(self, j, v):

        if issparse(self.matrix):
            for i in v.nonzero():
                self.matrix[i,j] = v[i]
        else:
            self.matrix[i] = v






class VectorSpaceModel(ModelBase):
    
    def __init__(self, corpus, corpus_param, model, model_param):
        
        ModelBase.__init__(self, corpus, corpus_param,
                           model, model_param)

        
        #TODO: Move this into Beagle
        self.dimension = 2048

        self.vector_path = os.path.join(self.model_path, 'vectors')

        #TODO: set up getter and setter correctly 
        self.stored_vectors = None





    def write_vector(self, vector, index):

        vector_name = '-'.join([self.corpus, self.corpus_param,
                                self.model, self.model_param, 
                                'vector-' + str(index) + '.pickle'])
        
        vector_file = os.path.join(self.vector_path, vector_name)

        # print 'Writing vector', index
        with open(vector_file, 'w') as f:
            pickle.dump(vector, f)

        return


    def vector(self, index):
        
        vector_name = '-'.join([self.corpus, self.corpus_param,
                                self.model, self.model_param, 
                                'vector-' + str(index) + '.pickle'])

        vector_file = os.path.join(self.vector_path, vector_name)

        # print 'Retrieving vector', index
        with open(vector_file, 'r') as f:
            return pickle.load(f)


    def write_vectors(self, vectors):
        
        print 'Writing vectors to', self.vector_path

        for i in xrange(len(vectors)):
            self.write_vector(vectors[i], i)

        return
            

    def vectors(self):

        if self.stored_vectors != None:

            return self.stored_vectors

        else:
        
            vector_files = os.listdir(self.vector_path)
        
            vectors = np.zeros((len(vector_files), self.dimension), 
                           dtype=np.float32)

            print 'Retrieving vectors from', self.vector_path

            sys.stdout.write('Retrieved vector ')
            progress = ''

            for i,vector_file in enumerate(vector_files):
                
                # Progress meter
                for char in progress:
                    sys.stdout.write('\b')
                progress = (str(i+1) + ' of ' + str(len(vector_files)))
                sys.stdout.write(progress)
                sys.stdout.flush()

                vectors[i] += self.vector(i)
            
            print 

            self.stored_vectors = vectors

            return vectors

    
    def compute_cosines(self, vector):
        
        cosine_fn.vector1 = vector

        print 'Computing a cosine similarity vector'

        p = Pool()
        similarity_vector = p.map(cosine_fn, self.vectors(), 5000)
        p.close()
        
        return np.array(similarity_vector, dtype=np.float32)


    def similar(self, query, n=-1, operator='avg_cosines',
                filter_stopwords = True, filter_degenerate = True):

        # TODO: User friendly error handling
        query_vectors = self.process_query(query)

        if operator == 'avg_cosines':
            
            sim_vectors = [self.compute_cosines(v) 
                           for v in query_vectors]
            similarity_vector = reduce(lambda w, v: w + v, sim_vectors)
            similarity_vector = similarity_vector / len(sim_vectors)

        elif operator == 'convolve':
            pass

        elif operator == 'sum_vectors':
            normed_vectors = [normalize(v) for v in query_vectors]
            vector_sum = reduce(lambda w, v: w + v, normed_vectors)
            similarity_vector = self.compute_cosines(vector_sum)

        else:
            raise Exception('Unrecognized operator name')

        
        pairs = zip(self.lexicon, similarity_vector)
        print 'Sorting results'
        pairs.sort(key=lambda p: p[1], reverse = True)
        
        if filter_degenerate:
            print 'Filtering degenerate vectors'
            pairs = filter(lambda p: p[1] != -2, pairs)

        if n != -1:
            pairs = pairs[:(n + len(self.stopwords))]

        if filter_stopwords:
            print 'Filtering stop words'
            pairs = filter(lambda p: p[0] not in self.stopwords, pairs)

        if n != -1:
            pairs = pairs[:n]

        return pairs


    def print_similar(self, word, n=20, operator='avg_cosines'):
    
        pairs = self.similar(word, n=n, operator=operator)

        # TODO: Make pretty printer robust
        print ''.join(['-' for i in xrange(38)])
        print '{0:^25}{1:^12}'.format('Word','Similarity')
        for w,v in pairs:
            print '{0:<25}{1:^12.3f}'.format(w,float(v))
        print ''.join(['-' for i in xrange(38)])
            
        return


    def process_query(self, query):
    
        bag_words = Tokenizer(self.corpus,
                              self.corpus_param).tok_sent(query)
        
        print 'Filtering stop words'
        bag_words = filter(lambda w: w not in self.stopwords,
                           bag_words)

        bag_indices = [self.lexicon.index(word) for word in bag_words]
        
        bag_vectors = [self.vector(i) for i in bag_indices]

        print 'Filtering zero vectors'
        for i in xrange(len(bag_words)):
            if not np.any(bag_vectors[i]):
                del bag_vectors[i]
                del bag_words[i]
        
        print 'Final word sequence: {0}'.format(', '.join(bag_words))

        return bag_vectors


def cosine_fn(vector2):

    return vector_cos(cosine_fn.vector1, vector2)
