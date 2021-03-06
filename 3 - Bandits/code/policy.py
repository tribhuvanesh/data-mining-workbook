#!/usr/bin/env python2.7

import numpy as np
from numpy.linalg import inv
import numpy.random

ALPHA = 0.2

art_dict  = None
num_articles = None
M = {}
M_inv = {}
b = {}
w = {}
prev_pred = (0, 0, []) # (timestamp < t >, predicted_article_id < y_t >, user_features < z_t >)

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    global art_dict
    global num_articles

    art_dict = art
    num_articles = len(art)

    for art_id in art_dict:
        M[art_id] = np.identity(6)
        M_inv[art_id] = np.identity(6)
        b[art_id] = np.zeros(6)
        w[art_id] = np.zeros(6)


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    if reward == -1:
        return

    y_t = prev_pred[1]
    z_t = prev_pred[2]

    M[y_t] = M[y_t] + np.outer(z_t, z_t)
    b[y_t] = b[y_t] + reward * z_t

    M_inv[y_t] = inv(M[y_t])
    w[y_t] = np.inner(M_inv[y_t], b[y_t])


# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global prev_pred

    ucb = {k:0 for k in articles}
    z = np.array(user_features)

    for x in articles:
        ucb[x] = np.inner(w[x], z) + ALPHA * np.sqrt( np.inner(z, np.inner((M_inv[x]), z)) )

    to_predict = max(articles, key=lambda k:ucb[k])
    prev_pred = (timestamp, to_predict, z)
    return to_predict
