# cvb0_sklearn

cvb0_inf.py contains an implementation of the CVB0 algorithm for inference on LDA models.

The computationally intense part of the algorithm is imported from the file cvb0_main.pyx.

gen_lda.py implements the generative model for LDA, useful for testing that implementations are correct.

First, compile the cython to C / C to machine code by running:

python setup.py build_ext --inplace

Then, test_cvb0_inf.py performs analysis of the new implementation and the existing implementation on sythetic data.
