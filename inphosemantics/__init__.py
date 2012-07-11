import os
import tempfile
import shutil
import pickle
import cPickle
import bz2

import numpy as np

# from scipy.sparse import lil_matrix
# from scipy.io import mmwrite, mmread


def load_picklez(filename):
    """
    load_picklez takes a filename and loads it as though it were
    a python pickle. If the file ends in '.bz2', attempts are made first
    to decompress it.
    
    Parameters
    ----------
    filename : String
        String indicating the path to the corpus file.
        (eg: '/var/data/abraham-lincoln.bz2').

    Returns
    -------
    unserialized Python object. See Python pickles.
        (http://docs.python.org/library/pickle.html)

    Examples
    --------
    
    >> inphosemantics.load_picklez('/var/data/abraham-lincoln.bz2')
    <president at 0x0001861>
    
    """
    
    print 'Loading corpus from \n'\
          '  ', filename
    
    if filename.endswith('.bz2'):

        f = bz2.BZ2File(filename, 'r')
        try:
            return pickle.loads(f.read())
        finally:
            f.close()
    
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def load_cPicklez(filename):
    """
    load_cPicklez takes a filename and loads it as though it were
    a python pickle. If the file ends in '.bz2', attempts are made first
    to decompress it. load_cPicklez is identical to load_picklez except
    that it uses cPickle over pickle for time optimization. Caveats on
    cPickle can be found at
    http://docs.python.org/library/pickle.html#module-cPickle .
    
    Parameters
    ----------
    filename : String
        String indicating the path to the corpus file.
        (eg: '/var/data/abraham-lincoln.bz2').

    Returns
    -------
    unserialized Python object. See Python pickles.
        (http://docs.python.org/library/pickle.html)

    Examples
    --------
    
    >> inphosemantics.load_cPicklez('/var/data/abraham-lincoln.bz2')
    <president at 0x0001861>
    
    """
    
    print 'Loading corpus from \n'\
          '  ', filename
    
    if filename.endswith('.bz2'):

        f = bz2.BZ2File(filename, 'r')
        try:
            return cPickle.loads(f.read())
        finally:
            f.close()
    
    else:
        with open(filename, 'rb') as f:
            return cPickle.load(f)




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
    """
    load_matrix takes a filename and loads it as though it were numpy
    matrix.
    
    Parameters
    ----------
    filename : String
        String indicating the path to the corpus file.
        (eg: '/var/data/abraham-lincoln.bz2').

    Returns
    -------
    unserialized Python object. See Python pickles.
        (http://docs.python.org/library/pickle.html)

    Examples
    --------
    
    >> inphosemantics.load_picklez('/var/data/abraham-lincoln.bz2')
    <president at 0x0001861>

    """

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
