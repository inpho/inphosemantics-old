from multiprocessing import Pool
import sys
import os
import pickle
import tempfile
import shutil
import numpy as np
from numpy.dual import fft, ifft

from inphosemantics.model import ModelBase
from inphosemantics.localmath\
    import init_gen_randvec, gen_randvec, RandomWordGen




class BeagleBase(ModelBase):

    def __init__(self, corpus, corpus_param, model_param):

        ModelBase.__init__(self, corpus, corpus_param, 
                           'beagle', model_param)

        self.dimension = 2048
        self.lmda = 7

        self.vector_path = os.path.join(self.model_path, 'vectors')

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

        if self.stored_vectors:

            return self.stored_vectors

        else:
        
            vector_files = os.listdir(self.vector_path)
        
            vectors = np.zeros((len(vector_files), self.dimension), 
                           dtype=np.float32)

            print 'Retrieving vectors from', self.vector_path

            for i,vector_file in enumerate(vector_files):
                vectors[i] += self.vector(i)

            return vectors

    
    # def write_vectors(self, vectors):
        
    #     vectors_name = '-'.join([self.corpus, self.corpus_param, self.model,
    #                              self.model_param, 'vectors.pickle'])
    #     vectors_file = os.path.join(self.model_path, vectors_name)

    #     print 'Writing vectors'
    #     with open(vectors_file, 'w') as f:
    #         pickle.dump(vectors, f)

    #     return

    
    # def vectors(self):

    #     vectors_name = '-'.join([self.corpus, self.corpus_param, self.model,
    #                              self.model_param, 'vectors.pickle'])
    #     vectors_file = os.path.join(self.model_path, vectors_name)

    #     print 'Reading vectors'
    #     with open(vectors_file, 'r') as f:
    #         return pickle.load(f)



class BeagleEnvironment(BeagleBase):

    def __init__(self, corpus, corpus_param):

        BeagleBase.__init__(self, corpus, corpus_param, 'environment')


    def gen_vectors(self):

        init_gen_randvec(self.dimension)

        p = Pool()
        vectors = p.map(gen_randvec, self.lexicon, 1000)
        p.close()

        print 'Environment vectors computed.'

        vectors = np.array(vectors)
        self.write_vectors(vectors)
        
        return



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

        #TODO: this path and others should be attributes of Corpus
        #instances
        tok_path = os.path.join(self.corpus_path, 'tokenized')

        docs = os.listdir(tok_path)
        
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

        # Encode stop words as indices in the lexicon
        order_fn.stopwords =\
            [order_fn.lexicon[word] for word in self.stopwords]

        # Retrieve environment vectors
        order_fn.envecs =\
            BeagleEnvironment(self.corpus, self.corpus_param).vectors()

        # Pass the context function a function to read tokenized
        # sentences
        order_fn.tokenized_sentences = self.tokenized_sentences

        order_fn.dimension = self.dimension
        order_fn.lmda = self.lmda

        order_fn.temp_dir = tempfile.mkdtemp()


        #TODO: this path and others should be attributes of Corpus
        #instances
        tok_path = os.path.join(self.corpus_path, 'tokenized')

        print 'Computing DFTs of environment vectors'
        order_fn.fftenvecs = map(fft, order_fn.envecs)
        
        order_fn.placeholder =\
            fft(RandomWordGen(d=self.dimension).make_rep(''))


        docs = os.listdir(tok_path)
        q = Pool()
        results = q.map(order_fn, docs, 2)
        q.close()

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
    stopwords = order_fn.stopwords
    envecs = order_fn.envecs
    dimension = order_fn.dimension
    lmda = order_fn.lmda
    fftenvecs = order_fn.fftenvecs
    placeholder = order_fn.placeholder
    temp_dir = order_fn.temp_dir
    
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
                
                fftvecseq = ([fftenvecs[w] for w in sent[left:i]]
                             + [placeholder]
                             + [fftenvecs[w] for w in sent[i+1:right]])

                ordvec = ifft(reduce(lambda v1,v2: v1*v2, fftvecseq))
                
                memvecs[word] += ordvec


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
