import os
import shutil
import tempfile
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.misc import factorial
from numpy.dual import fft, ifft

from inphosemantics import load_matrix, dump_matrix
from inphosemantics.model import Model
from inphosemantics.model.beagleenvironment import BeagleEnvironment



class RandomPermutations(object):
    

    def __init__(self, dimension, n, seed=None):

        if n > factorial(dimension):

            raise Exception('Maximum number of distinct permutations exceeded.')
        
        np.random.seed(seed)
        
        self.permutations = dict()

        _permutations = []

        for i in xrange(n):

            while True:

                idx_array = np.random.permutation(dimension)

                _idx_array = list(idx_array)

                if _idx_array not in _permutations:

                    _permutations.append(_idx_array)
                    
                    self.permutations[i] = self.mk_permutation(idx_array)

                    break
        


    @staticmethod
    def mk_permutation(idx_array):
        
        def p(vector):
            return vector[idx_array]

        return p
        




def order_fn(ind_sent_list):

    index = ind_sent_list[0]
    sent_list = ind_sent_list[1]


    env_matrix = order_fn.env_matrix
    lmbda = order_fn.lmbda
    temp_dir = order_fn.temp_dir
    left_permutation = order_fn.left_permutation
    right_permutation = order_fn.right_permutation
    placeholder = order_fn.placeholder


    mem_matrix = np.zeros(env_matrix.shape, dtype=np.float32)

    print 'Training on chunk of sentences', index

    for sent in sent_list:

        for k in xrange(1, lmbda):
            
            for i,word in enumerate(sent):
                
                a = i - k
                left = 0 if a < 0 else a
                b = i + k
                right = len(sent) if b > len(sent) else b
                
                vector_list = ([env_matrix[w] for w in sent[left:i]]
                               + [placeholder]
                               + [env_matrix[w] for w in sent[i+1:right]])


                def f(vector_list):

                    if len(vector_list) == 0:
                        return np.zeros(self.dimension)

                    elif len(vector_list) == 1:
                        return vector_list[0]

                    else:
                        v1 = fft(left_permutation(f(vector_list[:-1])))
                        v2 = fft(right_permutation(vector_list[len(vector_list)-1]))
                        return ifft(v1 * v2)


                order_vector = f(vector_list)
                mem_matrix[word] += order_vector


    print 'Chunk of sentences', index, '\n', mem_matrix
                    
    tmp_file =\
        os.path.join(temp_dir, 'order-' + str(index) + '.tmp.npy')

    print 'Dumping to temp file\n'\
          '  ', tmp_file
    
    dump_matrix(mem_matrix, tmp_file)
    
    return tmp_file





class BeagleOrder(Model):

    def train(self,
              corpus,
              token_type='sentences',
              stoplist=None,
              n_columns=None,
              env_matrix=None,
              placeholder=None,
              right_permutation=None,
              left_permutation=None,
              lmbda=7):


        if env_matrix == None:
            env_model = BeagleEnvironment()
            env_model.train(corpus,
                            token_type,
                            stoplist,
                            n_columns)
            env_matrix = env_model.matrix

        #For efficiency
        env_matrix = np.float32(env_matrix)

        order_fn.env_matrix = env_matrix

        
        temp_dir = tempfile.mkdtemp()
        order_fn.temp_dir = temp_dir


        order_fn.lmbda = lmbda


        if not placeholder:

            placeholder = np.random.random(env_matrix.shape[1])
            placeholder *= 2
            placeholder -= 1
            placeholder /= np.sum(placeholder**2)**(1./2)
            order_fn.placeholder = placeholder
                
        print 'Placeholder:', order_fn.placeholder
        print 'Norm of placeholder', np.sum(order_fn.placeholder**2)**(1./2)



        if not right_permutation or not left_permutation:
            permutations = RandomPermutations(env_matrix.shape[1], 2)

        if right_permutation:
            order_fn.right_permutation = right_permutation
        else:
            order_fn.right_permutation = permutations.permutations[0]

        if left_permutation:
            order_fn.left_permutation = left_permutation
        else:
            order_fn.left_permutation = permutations.permutations[1]

        print 'Right permutation', order_fn.right_permutation(np.arange(env_matrix.shape[1]))

        print 'Left permutation', order_fn.left_permutation(np.arange(env_matrix.shape[1]))




        n_cores = cpu_count()
        sentences = corpus.view_tokens(token_type)

        m = len(sentences) / n_cores
        sent_lists = [sentences[i*m:(i+1)*m]
                      for i in xrange(n_cores)]
        sent_lists[-1].extend(sentences[m*n_cores:])

        ind_sent_lists = list(enumerate(sent_lists))


        # Map
        p = Pool()
        results = p.map(order_fn, ind_sent_lists, 1)
        p.close()


        # Reduce
        self.matrix = np.zeros(env_matrix.shape, dtype=np.float32)
        
        for result in results:

            print 'Reducing', result

            summand = load_matrix(result)
            self.matrix += summand


        # Clean up
        print 'Deleting temporary directory\n'\
              '  ', temp_dir

        shutil.rmtree(temp_dir)




def test_BeagleOrder_1():

    import numpy as np
    from inphosemantics.corpus import BaseCorpus

    env_matrix = np.array([[2.,2.],[3.,3.]])

    c = BaseCorpus([0, 1, 1, 0, 1, 0, 0, 1, 1],
                   {'sentences': [2, 5]})

    m = BeagleOrder()
    m.train(c, env_matrix=env_matrix)

    print c.view_tokens('sentences')
    print m.matrix

    

def test_BeagleOrder_2():

    from inphosemantics import load_picklez, dump_matrix

    root = 'test-data/iep/plato/'

    corpus_filename =\
        root + 'corpus/iep-plato.pickle.bz2'

    env_filename =\
        root + 'models/iep-plato-beagleenviroment-sentences.npy'

    matrix_filename =\
        root + 'models/iep-plato-beagleorder-sentences.npy'


    print 'Loading corpus\n'\
          '  ', corpus_filename
    c = load_picklez(corpus_filename)
    

    print 'Loading environment model\n'\
          '  ', env_filename
    e = BeagleEnvironment()
    e.load_matrix(env_filename)
    print e.matrix

    print 'Training model'
    m = BeagleOrder()
    m.train(c, env_matrix=e.matrix)
    print m.matrix


    print 'Dumping matrix to\n'\
          '  ', matrix_filename
    m.dump_matrix(matrix_filename)
    
    return m
