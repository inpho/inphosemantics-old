from multiprocessing import Pool
import sys
import os
import pickle
import tempfile
import shutil
import numpy as np
from scipy.sparse import lil_matrix
from sparsesvd import sparsesvd

from math import log

from inphosemantics.model.vectorspacemodel import VectorSpaceModel

class Lsa(VectorSpaceModel):

    def __init__(self, corpus, corpus_param, model_param):
        
        VectorSpaceModel.__init__(self, corpus, corpus_param,
                                  'lsa', model_param)

        self.set_document_dict()
        self.set_n_documents()
        

    def get_document_dict(self):
        return self._document_dict

    def set_document_dict(self):
        
        # the keys in the document list are indices. The values are
        # pairs: (name corresponding to file, index within file)

        doc_list = []
        
        for name in os.listdir(self.tokenized_path):
            for i,paragraph in\
                    enumerate(self.tokenized_paragraphs(name)):                
                doc_list.append((name, i))

        self._document_dict = dict(zip(xrange(len(doc_list)), doc_list))
        
        return

    document_dict = property(get_document_dict, set_document_dict)


    def get_n_documents(self):
        
        return self._n_documents

        
    def set_n_documents(self):

        self._n_documents = len(self.document_dict)

        return

    n_documents = property(get_n_documents, set_n_documents)


    # Each vector is a column vector corresponding to a paragraph

    # Pass 1: perform term-document counts and store vectors

    # Pass 2: compute tfidfs and replace vectors

    # Pass 3: perform svd and replace vectors

    def first_pass(self):

        docs = os.listdir(self.tokenized_path)

        first_pass_fn.tokenized_paragraphs = self.tokenized_paragraphs
        first_pass_fn.lexicon = \
            dict(zip(self.lexicon, xrange(len(self.lexicon))))
        
        rev_document_alist =\
            [(val,key) for (key,val) in self.document_dict.items()]
        first_pass_fn.rev_document_dict = dict(rev_document_alist)
        
        first_pass_fn.write_vector = self.write_vector

        p = Pool()
        p.map(first_pass_fn, docs, 2)
        p.close()

        return


    def second_pass(self):
        
        df = np.zeros(len(self.lexicon))

        for i in xrange(self.n_documents):
            for term_index in self.vector(i):
                df[term_index] += 1
                
        for i in xrange(self.n_documents):
            
            sparse_vector = self.vector(i)
            
            for index,value in sparse_vector.iteritems():
            
                sparse_vector[index] *= log(self.n_documents / df[index])
                sparse_vector[index] = np.float32(sparse_vector[index])

            self.write_vector(sparse_vector, i)
            
        return

    def third_pass(self):

        # Assemble sparse matrix

        td_matrix = lil_matrix((len(self.lexicon), self.n_documents),
                                   dtype=np.float32)
        
        for col in xrange(self.n_documents):
            
            sparse_vector = self.vector(col)
            
            for row,value in sparse_vector.iteritems():

                # print 'Matrix shape:', td_matrix.shape
                # print 'Current index:', row, col

                td_matrix[row, col] = value

        td_matrix = td_matrix.tocsc()
       
        td_matrix_file = os.path.join(self.model_path, 'td-matrix.pickle')

        with open(td_matrix_file, 'w') as f:
            
            pickle.dump(td_matrix, f)

        # Run sparse svd

        k = 500

        ut, s, vt = sparsesvd(td_matrix, 500)

        print 'term vectors shape:', ut.shape
        print 'eigenvalue matrix shape:', s.shape
        print 'document vectors shape:', vt.shape
        
        #TODO Organize vectors in directories, etc.


        return
        


    def gen_vectors(self):
        
        self.first_pass()
        self.second_pass()

        return


def first_pass_fn(name):
    
    tokenized_paragraphs = first_pass_fn.tokenized_paragraphs
    lexicon = first_pass_fn.lexicon
    rev_document_dict = first_pass_fn.rev_document_dict
    write_vector = first_pass_fn.write_vector
    
    paragraphs = tokenized_paragraphs(name)

    for i,para in enumerate(paragraphs):
        
        # Homebrew sparse vector: dictionary with indices as keys and
        # vector components as values
        sparse_vector = dict()
        
        for word in para:
            if lexicon[word] in sparse_vector:
                sparse_vector[lexicon[word]] += 1
            else:
                sparse_vector[lexicon[word]] = 1

        write_vector(sparse_vector, rev_document_dict[(name, i)])

    return
    
    
