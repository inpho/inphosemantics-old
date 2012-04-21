import os.path
import sys
import pickle
from multiprocessing import Pool
import numpy as np
from numpy.dual import fft, ifft

from inphosemantics.model import ModelBase
from inphosemantics.corpus import Corpus
from inphosemantics.corpus.tokenizer import Tokenizer
from inphosemantics.localmath import vector_cos, normalize


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


    def similar(self, query, n=-1, 
                filter_stopwords = True, filter_degenerate = True):

        # TODO: User friendly error handling
        query_vector = self.process_query(query)
        similarity_vector = self.compute_cosines(query_vector)
        
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


    def print_similar(self, word, n=20):
    
        pairs = self.similar(word, n=n)

        # TODO: Make pretty printer robust
        print ''.join(['-' for i in xrange(38)])
        print '{0:^25}{1:^12}'.format('Word','Similarity')
        for w,v in pairs:
            print '{0:<25}{1:^12.3f}'.format(w,float(v))
        print ''.join(['-' for i in xrange(38)])
            
        return


    # TODO: Add holographic compression
    def process_query(self, query, convolve=False):
    
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

        normed_vectors = [normalize(v) for v in bag_vectors]

        if convolve:
            pass
        else:
            print 'Results based on vector sum (\'bag of words\').'
            result = reduce(lambda w, v: w + v, normed_vectors)

        return result


def cosine_fn(vector2):

    return vector_cos(cosine_fn.vector1, vector2)
