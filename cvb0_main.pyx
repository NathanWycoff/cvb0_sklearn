cimport numpy as np
import numpy as np
import array
from cpython cimport array

" In place modification of params"
def main_loop(docs, GAMMA, np.ndarray[np.float64_t, ndim = 2] Nwk, 
        np.ndarray[np.float64_t, ndim = 2] Nmk, 
        np.ndarray[np.float64_t, ndim = 1] Nk, double[:] eta, 
        double[:] alpha, int M, int K):

    cdef double eta_sum = sum(eta)

    cdef int[:] Ns = array.array('i', [len(x) for x in docs])

    cdef int m, n, w, k
    cdef double first_term, second_term, gam_sum
    cdef np.ndarray[np.float64_t, ndim = 2] GAMM

    for m in range(M):
        GAMM = GAMMA[m]
        for n in range(Ns[m]):
            w = docs[m][n]
            gam_sum = 0
            for k in range(K):
                #Remove the current val from consideration
                Nwk[w, k] -= GAMM[n, k]
                Nmk[m, k] -= GAMM[n, k]
                Nk[k] -= GAMM[n, k]

                #Calculate something propto the new val
                first_term = (Nwk[w, k] + eta[w]) / (Nk[k] + eta_sum)
                second_term = Nmk[m, k] + alpha[k]
                GAMM[n, k] = first_term * second_term
                gam_sum += GAMM[n, k]

            #Normalize GAMMA
            GAMM[n,] /= gam_sum

            #Update the counts
            for k in range(K):
                Nwk[w, k] += GAMM[n, k]
                Nmk[m, k] += GAMM[n, k]
                Nk[k] += GAMM[n, k]
