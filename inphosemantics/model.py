import os.path
import pickle

from inphosemantics.corpus import Corpus


class ModelBase(Corpus):

    def __init__(self, corpus, corpus_param, model, model_param):

        Corpus.__init__(self, corpus, corpus_param)

        self.model = model
        self.model_param = model_param

        self.model_path =\
            os.path.join(Corpus.data_root, self.corpus, self.corpus_param,
                         self.model, self.model_param)


class Model(ModelBase):

    def __init__(self, corpus, corpus_param, model, model_param):

        ModelBase.__init__(self, corpus, corpus_param, model, model_param)


    def sim_vec(self, index):
        
        filename = '-'.join([self.corpus, self.corpus_param,
                             self.model, self.model_param,
                             'cosines-' + str(index) + '.pickle'])
        filename = os.path.join(self.model_path, 'cosines', filename)
        
        print 'Reading similarity vector', filename
        with open(filename, mode='r') as f:
            return pickle.load(f)


    def similar(self, word, n=-1, 
                filter_stopwords = True, filter_degenerate = True):

        # TODO: User friendly error handling
        i = self.lexicon.index(word)
        vec = self.sim_vec(i)
        
        pairs = zip(self.lexicon, vec)
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

    # TODO: Finish this
    def parse_query(query):
    
        # bag = tok_sent(query)
        
        # print 'Filtering stop words'
        # bag = filter(lambda w: w not in stopwords, bag)
        
        # print 'Final bag of words: {0}'.format(', '.join(bag)) 
    
        return

