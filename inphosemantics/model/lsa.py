from multiprocessing import Pool
import sys
import os
import pickle
import tempfile
import shutil
import numpy as np

from inphosemantics.model.vectorspacemodel import VectorSpaceModel

class Lsa(VectorSpaceModel):

    def __init__(self, corpus, corpus_param, model_param):
        
        VectorSpaceModel.__init__(self, corpus, corpus_param,
                                  'lsa', model_param)

    # count the paragraphs in all the documents
    def compute_dimension(self):

        paragraph_count = 0

        for name in os.listdir(self.tokenized_path):
            paragraph_count += len(self.tokenized_paragraphs(name))

        return paragraph_count


    def gen_vectors(self):
        
        pass
