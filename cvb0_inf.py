#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  cvb0_inf.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.12.2018

import numpy as np
import array
from cvb0_main import main_loop

def cvb0_inf(docs, K, V, eta, alpha, max_iters = 100, thresh = 1e-2):
    Ns = array.array('i', [len(x) for x in docs])
    M = len(docs)

    # Randomly Initialize GAMMA, the variational parameter for Z, a ragged array which tells us which topic each word came from (and the only parameter not integrated out in collapsed VB).
    doc_inds = np.insert(np.cumsum(Ns), 0, 0).astype(int)
    #np.random.seed(123)
    #gamma = np.random.rand(sum(Ns) * K)
    GAMMA = [np.random.rand(Ns[m],K) for m in range(M)]
    #GAMMA = [np.array(GAM_init[(K*doc_inds[i]):(K*doc_inds[i+1])]).reshape([Ns[i],K]) for i in range(M)]
    GAMMA = [g / np.sum(g, 1)[:,np.newaxis] for g in GAMMA]#Rows should sum to 1.
    #GAMMA = np.asarray(GAMMA)

    # Initialize Nwk, which counts how many words of type v are in topic k,
    # Nmk, which counts how many words in topic k are in document m.
    # and Nk, how many words are in topic k.
    Nwk = np.zeros([V, K])
    Nmk = np.zeros([M, K])
    Nk = np.zeros([K])
    for k in range(K):
        for m in range(M):
            for n in range(Ns[m]):
                w = docs[m][n]
                Nwk[w, k] += GAMMA[m][n, k]
                Nmk[m, k] += GAMMA[m][n,k]
            Nk[k] += Nmk[m, k]

    it = 0
    diff = np.inf
    while it < max_iters and diff > thresh:
        it += 1
        main_loop(docs, GAMMA, Nwk, Nmk, Nk, eta, alpha, M, K)


    # Get point estimates of PHI, the topic by vocab matrix, and THETA, the document by topic matrix.
    PHI_hat = (Nwk / np.sum(Nwk, 0)[np.newaxis,:]).T
    THETA_hat = Nmk / np.sum(Nmk, 1)[:,np.newaxis]

    return({'PHI_hat' : PHI_hat,
        'THETA_hat' : THETA_hat})
