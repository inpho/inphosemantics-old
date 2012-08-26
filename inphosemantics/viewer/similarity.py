import numpy as np
from scipy import sparse



def_submat_size = 1e5



# TODO: Suppress division by zero errors; be sure that it's safe to do
# so.



def row_norms(matrix):

    norms = np.empty(matrix.shape[0])

    sp = sparse.issparse(matrix)

    for i in xrange(norms.shape[0]):

        row = matrix[i:i+1, :]

        if sp:
            
            row = row.toarray()

        norms[i] = np.dot(row, row.T)**0.5
        
    return norms



def similar_rows(row_index, matrix,
                 filter_nan=False,
                 norms=None,
                 submat_size=def_submat_size):
    """
    """
    if sparse.issparse(matrix):

        nums = sparse_mvdot(matrix, matrix[row_index:row_index+1, :].T,
                            submat_size=submat_size)

    else:
        
        nums = np.dot(matrix, matrix[row_index:row_index+1, :].T)

        nums = np.squeeze(nums)

    try:

        dens = norms * norms[row_index]

    except:

        norms = row_norms(matrix)

        dens = norms * norms[row_index]

    out = nums / dens

    return sort_sim(out, filter_nan=filter_nan)



def sparse_mvdot(m, v, submat_size=def_submat_size):
    """
    For sparse matrices. The assumption is that a dense view of the
    entire matrix is too large. So the matrix is split into
    submatrices (horizontal slices) which are converted to dense
    matrices one at a time.
    """

    m = m.tocsr()

    if sparse.issparse(v):

        v = v.toarray()

    v = v.reshape((v.size, 1))

    out = np.empty((m.shape[0], 1))

    if submat_size < m.shape[1]:

        print 'Note: specified submatrix size is'\
              'less than the number of columns in matrix'

        m_rows = 1

        k_submats = m.shape[0]

    else:

        m_rows = int(submat_size / m.shape[1])

        k_submats = int(m.shape[0] / m_rows)

    for i in xrange(k_submats):

        i *= m_rows

        j = i + m_rows

        submat = m[i:j, :]

        out[i:j, :] = np.dot(submat.toarray(), v)

    if j < m.shape[0]:

        submat = m[j:, :]

        out[j:, :] = np.dot(submat.toarray(), v)

    return np.squeeze(out)




def sort_sim(results, filter_nan=False):
    """
    """
    results = zip(xrange(results.shape[0]), results.tolist())
    
    # Filter out undefined results

    if filter_nan:
        results = [(i,v) for i,v in results if np.isfinite(v)]
        
    dtype = [('index', np.int), ('value', np.float)]

    # NB: numpy >= 1.4 sorts NaN to the end

    results = np.array(results, dtype=dtype)

    results.sort(order='value')

    results = results[::-1]

    return results



def similar_columns(column, matrix, filter_nan=False):
    """
    """
    return similar_rows(column, matrix.T, filter_nan=filter_nan)



def simmat_rows(matrix, row_indices):
    """
    """
    sim_matrix = SimilarityMatrix(row_indices)
    
    sim_matrix.compute(matrix)

    return sim_matrix



def simmat_columns(matrix, column_indices):
    """
    """
    sim_matrix = SimilarityMatrix(column_indices)

    sim_matrix.compute(matrix.T)

    return sim_matrix




# TODO: Compress symmetric similarity matrix. Cf. scipy.spatial.distance.squareform
class SimilarityMatrix(object):

    def __init__(self, indices=None, matrix=None):

        self.indices = indices

        if matrix == None:
            self.matrix = np.zeros((len(self.indices), len(self.indices)))


    def compute(self, data_matrix):
        """
        Comparisons are row-wise
        """

        if sparse.issparse(data_matrix):

            row_fn.matrix = np.zeros((len(self.indices), data_matrix.shape[1]))

            for i in xrange(len(self.indices)):
                row_fn.matrix[i,:] = data_matrix[self.indices[i],:].toarray()

        else:
            row_fn.matrix = data_matrix[self.indices]
            

        for idx_sim,idx in enumerate(self.indices):

            row_fn.v = data_matrix[idx,:]
            if sparse.issparse(row_fn.v):
                row_fn.v = np.squeeze(row_fn.v.toarray())

            p = Pool()
            results = p.map(row_fn, range(idx_sim, len(self.indices)))
            p.close()

            values = zip(*results)[1]

            self.matrix[idx_sim,idx_sim:] = np.asarray(values)

        # Redundant representation of a symmetric matrix
        self.matrix += self.matrix.T - np.diag(np.diag(self.matrix))



