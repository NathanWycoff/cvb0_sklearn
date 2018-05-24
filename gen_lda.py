#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  gen_lda.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.12.2018

from scipy import stats
import numpy as np


def gen_lda(K, V, M, N_mu, eta, alpha):
    # Generate topics
    PHI = np.zeros([K,V])
    for k in range(K):
        PHI[k,] = stats.gamma.rvs(eta, size = V)
        PHI[k,] = PHI[k,] / sum(PHI[k,])

    # Generate docs
    THETA = np.zeros([M,K])
    docs = []
    for m in range(M):
        # Generate the proportion of each topic in each document.
        THETA[m,] = stats.gamma.rvs(alpha, size = K)
        THETA[m,] = THETA[m,] / sum(THETA[m,])

        # Generate the number of words in each doc
        N = stats.poisson.rvs(N_mu - 1) + 1

        docs.append([])
        # Sample each word.
        for n in range(N):
            z = np.argmax(np.random.multinomial(1, THETA[m,]))
            w = np.argmax(np.random.multinomial(1, PHI[z,]))

            docs[-1].append(w)

    return({'docs' : docs,
        'PHI' : PHI,
        'THETA' : THETA})
