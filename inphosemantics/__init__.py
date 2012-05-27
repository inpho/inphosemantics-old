import os
import tempfile
import shutil
import pickle
import bz2

import numpy as np

# from scipy.sparse import lil_matrix
# from scipy.io import mmwrite, mmread


def load_picklez(filename):
    """
    Takes a filename and loads it as though it were a python pickle.
    If the file ends in '.bz2', tries to decompress it first.
    """
    if filename.endswith('.bz2'):

        f = bz2.BZ2File(filename, 'r')
        try:
            return pickle.loads(f.read())
        finally:
            f.close()
    
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)




#TODO: Verify, handle filename extensions: should be either .mtx or .mtx.bz2

# def load_matrix(filename):

#     return mmread(filename)


# def dump_matrix(matrix, filename, **kwargs):

#     mmwrite(filename, matrix, **kwargs)



# def dump_matrixz(matrix, filename, **kwargs):

#     tmp_dir = tempfile.mkdtemp()
#     tmp = os.path.join(tmp_dir, 'tmp-file.mtx')

#     dump_matrix(matrix, tmp, **kwargs)
    
#     # Need to reopen tmp as mmwrite closed it
#     f = open(tmp, 'r')
#     out = bz2.BZ2File(filename, 'w')
#     try:
#         out.writelines(f)
#     finally:
#         out.close()
#         f.close()            
#         shutil.rmtree(tmp_dir)


def load_matrix(filename):

    # The slice [()] is for the cases where np.save has stored a
    # sparse matrix in a zero-dimensional array

    return np.load(filename)[()]


def dump_matrix(matrix, filename):

    np.save(filename, matrix)


# def dump_matrixz(matrix, filename):

#     if not filename.endswith('.bz2'):
#         filename = filename + '.bz2'

#     tmp_dir = tempfile.mkdtemp()
#     tmp = os.path.join(tmp_dir, 'tmp-file.npy')

#     dump_matrix(matrix, tmp)
    
#     # Need to reopen tmp as np.save closed it
#     f = open(tmp, 'r')
#     out = bz2.BZ2File(filename, 'w')
#     try:
#         out.writelines(f)
#     finally:
#         out.close()
#         f.close()            
#         shutil.rmtree(tmp_dir)
