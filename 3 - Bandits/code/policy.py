#!/usr/bin/env python2.7

import numpy as np
from numpy.linalg import inv
import numpy.random

ALPHA = 0.3

art_dict = None
num_articles = None
M = {}
M_inv = {}
b = {}
prev_pred = (0, 0, []) # (timestamp < t >, predicted_article_id < y_t >, user_features < z_t >)

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    global art_dict

    art_dict = art
    num_articles = len(art)

    for art_id in art:
        M[art_id] = np.identity(12)
        M_inv[art_id] = np.identity(12)
        b[art_id] = np.zeros(12)


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):

    y_t = prev_pred[1]
    z_t = prev_pred[2]

    M[y_t] = M[y_t] + np.outer(z_t, z_t)
    b[y_t] = b[y_t] + reward * z_t

    M_inv[y_t] = inv(M[y_t])


# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global prev_pred

    ucb = {k:0 for k in articles}
    z = {}
    for a in articles:
        z[a] = np.concatenate((user_features, art_dict[a]))

    for x in articles:
        w = np.inner(M_inv[x], b[x])
        ucb[x] = np.inner(w, z[x]) + ALPHA * np.sqrt( np.inner(z[x], np.inner((M_inv[x]), z[x])) )

    to_predict = max(articles, key=lambda k:ucb[k])
    prev_pred = (timestamp, to_predict, z[to_predict])
    return to_predict
