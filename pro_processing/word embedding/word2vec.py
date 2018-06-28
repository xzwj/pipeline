#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
from gensim.models.word2vec_inner import FAST_VERSION
from gensim.models.word2vec import Word2Vec, LineSentence

def word_embedding(train,output,window=5,size=100,sample=1e-3,hs=0,negative=5,
                   threads=12,iter=5,min_count=5,cbow=1,binary=0):
    """
    :param train: Use text data from file TRAIN to train the model, required=True
    :param output: Use file OUTPUT to save the resulting word vectors
    :param window: Set max skip length WINDOW between words; default is 5", type=int, default=5
    :param size: Set size of word vectors; default is 100", type=int, default=100
    :param sample:
    "Set threshold for occurrence of words. "
    #          "Those that appear with higher frequency in the training data will be randomly down-sampled;"
    #          " default is 1e-3, useful range is (0, 1e-5)",
    #     type=float, default=1e-3
    :param hs:
    help="Use Hierarchical Softmax; default is 0 (not used)",
    #     type=int, default=0, choices=[0, 1]
    :param negative:
    "Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)",
    #     type=int, default=5
    :param threads: "Use THREADS threads (default 12)", type=int, default=12
    :param iter: "Run more training iterations (default 5)", type=int, default=5
    :param min_count:
    "This will discard words that appear less than MIN_COUNT times; default is 5",
    #     type=int, default=5
    :param cbow:
    "Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)",
    #     type=int, default=1, choices=[0, 1]
    :param binary:
    "Save the resulting vectors in binary mode; default is 0 (off)",
    #     type=int, default=0, choices=[0, 1]
    :param accuracy: Use questions from file ACCURACY to evaluate the model
    :return:
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )
    logger.info("running %s", " ".join(sys.argv))
    logger.info("using optimization %s", FAST_VERSION)

    if cbow == 0:
        skipgram = 1
    else:
        skipgram = 0

    corpus = LineSentence(train)

    model = Word2Vec(
        corpus, size=size, min_count=min_count, workers=threads,
        window=window, sample=sample, sg=skipgram, hs=hs,
        negative=negative, cbow_mean=1, iter=iter
    )

    if output:
        outfile = output
        model.wv.save_word2vec_format(outfile, binary=binary)
        model.save(outfile + '.model')
    else:
        outfile = train
        model.save(outfile + '.model')

    logger.info("finished running %s", " ".join(sys.argv))

if __name__=="__main__":
    word_embedding('text8','output')