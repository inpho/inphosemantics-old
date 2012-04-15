from multiprocessing import Pool
import sys
import os
import pickle
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


    def write_vector(self, vector):
        pass

    
    def write_vectors(self, vectors):
        
        vectors_name = '-'.join([self.corpus, self.corpus_param, self.model,
                                 self.model_param, 'vectors.pickle'])
        vectors_file = os.path.join(self.model_path, vectors_name)

        print 'Writing vectors'
        with open(vectors_file, 'w') as f:
            pickle.dump(vectors, f)
            return


    def vector(self):
        pass

    
    def vectors(self):

        vectors_name = '-'.join([self.corpus, self.corpus_param, self.model,
                                 self.model_param, 'vectors.pickle'])
        vectors_file = os.path.join(self.model_path, vectors_name)

        print 'Reading vectors'
        with open(vectors_file, 'r') as f:
            return pickle.load(f)



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
        doc_context.lexicon =\
            dict(zip(self.lexicon, xrange(len(self.lexicon))))

        # Encode stop words as indices in the lexicon
        doc_context.stopwords =\
            [doc_context.lexicon[word] for word in self.stopwords]

        # Retrieve environment vectors
        doc_context.envecs =\
            BeagleEnvironment(self.corpus, self.corpus_param).vectors()

        # Pass the context function a function to read tokenized
        # sentences
        doc_context.tokenized_sentences = self.tokenized_sentences

        doc_context.dimension = self.dimension

        #TODO: this path and others should be attributes of Corpus
        #instances
        tok_path = os.path.join(self.corpus_path, 'tokenized')

        docs = os.listdir(tok_path)

        memvecs = np.zeros((len(self.lexicon), self.dimension))
        
        p = Pool()
        results = p.map(doc_context, docs, 2)
        p.close()

        # Reduce
        for result in results:
            for word,vec in result.iteritems():
                memvecs[word] = memvecs[word] + vec
        
        self.write_vectors(memvecs)
        
        return


# A natural place for this function would be in the BeagleContext
# class; however, multiprocessing does not allow for this. Here's a
# work around.
def doc_context(name):

    tokenized_sentences = doc_context.tokenized_sentences
    lexicon = doc_context.lexicon
    stopwords = doc_context.stopwords
    envecs = doc_context.envecs
    dimension = doc_context.dimension

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
                    
    print 'Processed', name

    return memvecs



class BeagleOrder(BeagleBase):

    def __init__(self, corpus, corpus_param):

        BeagleBase.__init__(self, corpus, corpus_param, 'order')
        
    def gen_vecs(self):

        # Make lexicon a dictionary for rapid reverse look-up
        doc_order.lexicon =\
            dict(zip(self.lexicon, xrange(len(self.lexicon))))

        # Encode stop words as indices in the lexicon
        doc_order.stopwords =\
            [doc_order.lexicon[word] for word in self.stopwords]

        # Retrieve environment vectors
        doc_order.envecs =\
            BeagleEnvironment(self.corpus, self.corpus_param).vectors()

        # Pass the context function a function to read tokenized
        # sentences
        doc_order.tokenized_sentences = self.tokenized_sentences

        doc_order.dimension = self.dimension

        #TODO: this path and others should be attributes of Corpus
        #instances
        tok_path = os.path.join(self.corpus_path, 'tokenized')

        print 'Computing DFTs of environment vectors'
        doc_order.fftenvecs = map(fft, envecs)
        
        doc_order.placeholder =\
            fft(RandomWordGen(d=dimension).make_rep(''))


        docs = os.listdir(tok_path)
        q = Pool()
        results = q.map(doc_order, docs, 2)
        q.close()


        # Reduce
        memvecs = np.zeros((len(self.lexicon), self.dimension))
        for result in results:
            for word,vec in result.iteritems():
                memvecs[word] = memvecs[word] + vec

        write_vecs(memvecs)
        return


# A natural place for this function would be in the BeagleOrder
# class; however, multiprocessing does not allow for this. Here's a
# work around.
def doc_order(name):

    tokenized_sentences = doc_order.tokenized_sentences
    lexicon = doc_order.lexicon
    stopwords = doc_order.stopwords
    envecs = doc_order.envecs
    dimension = doc_order.dimension
    fftenvecs = doc_order.fftenvecs
    placeholder = doc_order.placeholder
    
    sents = read_sentences(name)
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

    return memvecs



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
