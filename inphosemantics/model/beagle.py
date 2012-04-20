from multiprocessing import Pool
import sys
import os
import pickle
import tempfile
import shutil
import numpy as np
from numpy.dual import fft, ifft

from inphosemantics.model.vectorspacemodel import VectorSpaceModel
from inphosemantics.localmath\
    import RandomVectors, RandomPermutations




class BeagleBase(VectorSpaceModel):

    def __init__(self, corpus, corpus_param, model_param):

        VectorSpaceModel.__init__(self, corpus, corpus_param, 
                                  'beagle', model_param)

        self.lmda = 7


class BeagleEnvironment(BeagleBase):

    def __init__(self, corpus, corpus_param):

        BeagleBase.__init__(self, corpus, corpus_param, 'environment')


    def gen_vectors(self):

        environment_fn.meaning_vector =\
            RandomVectors(self.dimension).meaning_vector

        p = Pool()
        vectors = p.map(environment_fn, self.lexicon, 1000)
        p.close()

        print 'Environment vectors computed.'

        vectors = np.array(vectors)
        self.write_vectors(vectors)
        
        return



def environment_fn(word):

    return environment_fn.meaning_vector(word)


class BeagleContext(BeagleBase):

    def __init__(self, corpus, corpus_param):

        BeagleBase.__init__(self, corpus, corpus_param, 'context')
        

    def gen_vectors(self):

        # Make lexicon a dictionary for rapid reverse look-up
        context_fn.lexicon =\
            dict(zip(self.lexicon, xrange(len(self.lexicon))))

        # Encode stop words as indices in the lexicon
        context_fn.stopwords =\
            [context_fn.lexicon[word] for word in self.stopwords]

        # Retrieve environment vectors
        context_fn.envecs =\
            BeagleEnvironment(self.corpus, self.corpus_param).vectors()

        # Pass the context function a function to read tokenized
        # sentences
        context_fn.tokenized_sentences = self.tokenized_sentences

        context_fn.dimension = self.dimension

        context_fn.temp_dir = tempfile.mkdtemp()

        docs = os.listdir(self.tokenized_path)
        
        p = Pool()
        results = p.map(context_fn, docs, 2)
        p.close()

        # Reduce
        memvecs = np.zeros((len(self.lexicon), self.dimension))
        for tmp_file in results:

            print 'Reducing', tmp_file

            with open(tmp_file, 'r+') as f:
                result = pickle.load(f)

            for word,vec in result.iteritems():
                memvecs[word] = memvecs[word] + vec

        # Clean up
        shutil.rmtree(context_fn.temp_dir)
        
        self.write_vectors(memvecs)
        
        return


# A natural place for this function would be in the BeagleContext
# class; however, multiprocessing does not allow for this. Here's a
# work around.
def context_fn(name):

    tokenized_sentences = context_fn.tokenized_sentences
    lexicon = context_fn.lexicon
    stopwords = context_fn.stopwords
    envecs = context_fn.envecs
    dimension = context_fn.dimension
    temp_dir = context_fn.temp_dir

    sents = tokenized_sentences(name)
    def encode(sent):
        return [lexicon[word] for word in sent]
    sents = [encode(sent) for sent in sents]
    doclex = set([word for sent in sents for word in sent])
    zerovecs = [np.zeros(dimension) for i in xrange(len(doclex))]
    memvecs = dict(zip(doclex, zerovecs))
    
    for sent in sents:
        for i,word in enumerate(sent):
            context = sent[:i] + sent[i+1:]
            for ctxword in context:
                if ctxword not in stopwords:
                    memvecs[word] += envecs[ctxword]
                    
    tmp_file = os.path.join(temp_dir, 'context-' + name + '.tmp')
    with open(tmp_file, 'w') as f:
        pickle.dump(memvecs, f)

    return tmp_file




class BeagleOrder(BeagleBase):

    def __init__(self, corpus, corpus_param):

        BeagleBase.__init__(self, corpus, corpus_param, 'order')
        
    def gen_vectors(self):

        # Make lexicon a dictionary for rapid reverse look-up
        order_fn.lexicon =\
            dict(zip(self.lexicon, xrange(len(self.lexicon))))

        # Retrieve environment vectors
        order_fn.envecs =\
            BeagleEnvironment(self.corpus, self.corpus_param).vectors()

        # Pass the context function a function to read tokenized
        # sentences
        order_fn.tokenized_sentences = self.tokenized_sentences

        order_fn.dimension = self.dimension
        order_fn.lmda = self.lmda

        order_fn.temp_dir = tempfile.mkdtemp()

        order_fn.placeholder =\
            RandomVectors(self.dimension).meaning_vector('')

        permutations = RandomPermutations(self.dimension, 2).permutations
        order_fn.left_permutation = permutations[1]
        order_fn.right_permutation = permutations[2]


        docs = os.listdir(self.tokenized_path)

        q = Pool()
        results = q.map(order_fn, docs, 2)
        q.close()

        # results = map(order_fn, docs)

        # Reduce
        memvecs = np.zeros((len(self.lexicon), self.dimension))
        for tmp_file in results:

            print 'Reducing', tmp_file

            with open(tmp_file, 'r+') as f:
                result = pickle.load(f)

            for word,vec in result.iteritems():
                memvecs[word] = memvecs[word] + vec

        # Clean up
        shutil.rmtree(order_fn.temp_dir)

        self.write_vectors(memvecs)

        return



# A natural place for this function would be in the BeagleOrder
# class; however, multiprocessing does not allow for this. Here's a
# work around.
def order_fn(name):

    tokenized_sentences = order_fn.tokenized_sentences
    lexicon = order_fn.lexicon
    envecs = order_fn.envecs
    dimension = order_fn.dimension
    lmda = order_fn.lmda
    temp_dir = order_fn.temp_dir
    left_permutation = order_fn.left_permutation
    right_permutation = order_fn.right_permutation
    placeholder = order_fn.placeholder

    sents = tokenized_sentences(name)
    def encode(sent):
        return [lexicon[word] for word in sent]
    sents = [encode(sent) for sent in sents]
    doclex = set([word for sent in sents for word in sent])
    zerovecs = [np.zeros(dimension) for i in xrange(len(doclex))]
    memvecs = dict(zip(doclex, zerovecs))

    for sent in sents:
        for k in xrange(1, lmda):
            for i,word in enumerate(sent):
                
                a = i - k
                left = 0 if a < 0 else a
                b = i + k
                right = len(sent) if b > len(sent) else b
                
                vector_list = ([envecs[w] for w in sent[left:i]]
                               + [placeholder]
                               + [envecs[w] for w in sent[i+1:right]])

                def f(ls):
                    # The first case should not occur (empty sentence)
                    if len(ls) == 0:
                        return np.zeros(self.dimension)

                    elif len(ls) == 1:
                        return ls[0]

                    else:
                        v1 = fft(left_permutation(f(ls[:-1])))
                        v2 = fft(right_permutation(ls[len(ls)-1]))
                        return ifft(v1 * v2)


                order_vector = f(vector_list)
                memvecs[word] += order_vector


    tmp_file = os.path.join(temp_dir, 'order-' + name + '.tmp')
    with open(tmp_file, 'w') as f:
        pickle.dump(memvecs, f)

    return tmp_file




class BeagleComposite(BeagleBase):

    def __init__(self, corpus, corpus_param):

        BeagleBase.__init__(self, corpus, corpus_param, 'composite')


    def gen_vectors(self):

        context_vecs =\
            BeagleContext(self.corpus, self.corpus_param).vectors()
        order_vecs =\
            BeagleOrder(self.corpus, self.corpus_param).vectors()
        memory_vecs = context_vecs + order_vecs

        self.write_vectors(memory_vecs)
        
        return
