#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  test_cvb0_inf.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.12.2018

## Test that we recover the truth for LDA models.

from gen_lda import gen_lda
from cvb0_inf import cvb0_inf
import numpy as np
import time
import array


# Sim params
K = 30# Number of topics
V = 400# Size of vocab
M = 200# Number of documents
N_mu = 100# Mean number of words per doc

# Hyper params
eta = np.repeat(0.5, V) #Prior on vocab words in topics.
alpha = np.repeat(0.5, K) #Prior on topics in documents.

# Simulate a dataset from the LDA generative model
gen = gen_lda(K, V, M, N_mu, eta, alpha)

# Time our model
%time fit_py = cvb0_inf(gen['docs'], K, V, eta, alpha, max_iters = 100)
print fit_py['PHI_hat']
print gen['PHI']

# Get the data in the proper form for the built in func
tf = np.zeros([M,V])
Ns = [len(x) for x in gen['docs']]
for m in range(M):
    for n in range(Ns[m]):
        tf[m, gen['docs'][m][n]] += 1

# Try the built in method
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = K, learning_method = 'batch', max_iter = 100)
%time lda.fit(tf)

# Check accuracy
print gen['PHI']
print fit['PHI_hat']
