import tempfile
import shutil
import os
import bz2

from numpy import matrix
from scipy.sparse import issparse, lil_matrix
from scipy.io import mmwrite, mmread


#TODO: Verify, handle filename extensions: should be either .mtx or .mtx.bz2

    

def load_matrix(filename):

    m = mmread(filename)
        
    if issparse(m):
        return SparseMatrix(m)
    else:
        return DenseMatrix(m)

    

class SparseMatrix(lil_matrix):

    def __init__(self, *args, **kwargs):

        super(SparseMatrix, self).__init__(*args, **kwargs)
            
        # This is needed to get the right format name
        self.format = 'lil'
        

    #TODO give an informative comment about the data file
    def dump(self, filename, **kwargs):

        mmwrite(filename, self.tocsr(), **kwargs)


    def dumpz(self, filename, **kwargs):

        tmp_dir = tempfile.mkdtemp()
        tmp = os.path.join(tmp_dir, 'tmp-file.mtx')

        mmwrite(tmp, self.tocsr(), **kwargs)

        # Need to reopen tmp as mmwrite closed it
        f = open(tmp, 'r')
        out = bz2.BZ2File(filename, 'w')
        try:
            out.writelines(f)
        finally:
            out.close()
            f.close()            
            shutil.rmtree(tmp_dir)



class DenseMatrix(matrix):
    """
    For an explanation of the mechanics of this subclass, see how to
    subclass numpy ndarrays:
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing
    """

    #TODO See if this can be omitted entirely since no new attributes
    #are needed
    def __new__(subtype, data, dtype=None, copy=True):

        obj = matrix.__new__(subtype, data, dtype, copy)
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

                                            
    #TODO give an informative comment about the data file
    def dump(self, filename, **kwargs):

        mmwrite(filename, self, **kwargs)


    def dumpz(self, filename, **kwargs):

        tmp_dir = tempfile.mkdtemp()
        tmp = os.path.join(tmp_dir, filename)

        mmwrite(tmp, self, **kwargs)

        # Need to reopen tmp as mmwrite closed it
        f = open(tmp, 'r')
        out = bz2.BZ2File(filename, 'w')
        try:
            out.writelines(f)
        finally:
            out.close()
            f.close()            
            shutil.rmtree(tmp_dir)

