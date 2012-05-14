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
        return SparseMatrix(m, filename = filename)
    else:
        return DenseMatrix(m, filename = filename)

    

class SparseMatrix(lil_matrix):

    def __init__(self, *args, **kwargs):

        if 'filename' in kwargs:
            self.filename = kwargs['filename']
            del kwargs['filename']
        else:
            self.filename = ''

        super(SparseMatrix, self).__init__(*args, **kwargs)
            
        # This is needed to get the right format name
        self.format = 'lil'
        
        
    def __repr__(self):

        base_msg = super(SparseMatrix, self).__repr__()[1:-1]
        ext_msg = "<%s\n"\
                  "\tfilename = '%s'>" %\
                  (base_msg, self.filename)
        return ext_msg

    #TODO give an informative comment about the data file
    def dump(self, **kwargs):

        mmwrite(self.filename, self.tocsr(), **kwargs)


    def dumpz(self, **kwargs):

        tmp_dir = tempfile.mkdtemp()
        tmp = os.path.join(tmp_dir, self.filename)

        mmwrite(tmp, self.tocsr(), **kwargs)

        # Need to reopen tmp as mmwrite closed it
        f = open(tmp, 'r')
        out = bz2.BZ2File(self.filename + '.bz2', 'w')
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
    
    def __new__(subtype, data, dtype=None, copy=True, filename=''):

        obj = matrix.__new__(subtype, data, dtype, copy)
        obj.filename = filename
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.filename = getattr(obj, 'filename', None)
                                            
    #TODO give an informative comment about the data file
    def dump(self, **kwargs):

        mmwrite(self.filename, self, **kwargs)


    def dumpz(self, **kwargs):

        tmp_dir = tempfile.mkdtemp()
        tmp = os.path.join(tmp_dir, self.filename)

        mmwrite(tmp, self, **kwargs)

        # Need to reopen tmp as mmwrite closed it
        f = open(tmp, 'r')
        out = bz2.BZ2File(self.filename + '.bz2', 'w')
        try:
            out.writelines(f)
        finally:
            out.close()
            f.close()            
            shutil.rmtree(tmp_dir)

