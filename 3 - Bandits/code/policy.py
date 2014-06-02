#!/usr/bin/env python2.7

import numpy as np
from numpy.linalg import inv
import numpy.random

ALPHA = 0.40
k = 36
d = 6

art_dict  = None
num_articles = None
A = {}
A_inv = {}
B = {}
b = {}
theta = {}
prev_pred = (0, 0, [], []) # (timestamp < t >, predicted_article_id < y_t >, user_features < x_t >, user_art_features < z_t >)

A_0 = np.identity(k)
A_0_inv = np.identity(k)
b_0 = np.zeros(k)
beta = np.dot(A_0_inv, b_0)

tmpk1 = np.zeros(k)
tmp1d = np.zeros(d)

# print np.shape(A_0_inv)
# print np.shape(b_0)
# print np.shape(beta)

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    global art_dict
    global num_articles

    art_dict = {}
    for key in art:
        art_dict[key] = np.array(art[key])
    num_articles = len(art)

    for art_id in art_dict:
        A[art_id] = np.identity(d)
        B[art_id] = np.zeros((d, k))
        b[art_id] = np.zeros(d)
        A_inv[art_id] = np.identity(d)
        theta[art_id] = np.zeros(d)


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global beta, A_0, b_0, A_0_inv
    if reward == -1:
        return

    a = prev_pred[1]
    x = prev_pred[2]
    z = prev_pred[3]

    tmp_prod1 = np.dot(B[a].T, np.dot(A_inv[a], B[a]))
    tmp_prod2 = np.dot(B[a].T, np.dot(A_inv[a], b[a]))

    A_0 = A_0 + tmp_prod1
    b_0 = b_0 + tmp_prod2
    A[a] = A[a] + np.outer(x, x)
    B[a] = B[a] + np.outer(x, z)
    b[a] = b[a] + reward*x
    A_0 = A_0 + np.outer(z, z) - tmp_prod1
    b_0 = b_0 + reward*z - tmp_prod2

    # Update cached variables
    A_0_inv = inv(A_0)
    A_inv[a] = inv(A[a])

    theta[a] = np.dot( A_inv[a],
                         (b[a] - np.dot(B[a], beta))
                       )
    beta = np.dot(A_0_inv, b_0)
    # print '----'
    # print np.shape(A_0_inv)
    # print np.shape(b_0)
    # print np.shape(beta)
    # print np.shape(tmp_prod2)


# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global prev_pred, tmpk1, tmp1d

    x_ta_dct = {}
    p_ta_dct = {}
    z_ta_dct = {}

    for a in articles:
        # x_ta : user features
        x_ta = np.array(art_dict[a])
        # z_ta : outer product of the user and article features
        z_ta = np.outer(np.array(user_features), x_ta).flatten()

        tmp_prod1 = np.dot(A_0_inv, np.dot(B[a].T, np.dot(A_inv[a], x_ta)), out=tmpk1) # Should be k x 1
        # print '---'
        # print np.shape(tmp_prod1), np.shape(tmpk1)
        tmp_prod2 = np.dot(x_ta.T, A_inv[a], out=tmp1d) # Should be 1 x d
        # print np.shape(tmp_prod2), np.shape(tmp1d)
        s_ta = np.dot(z_ta.T, np.dot(A_0_inv, z_ta)) \
               - 2 * np.dot(z_ta.T, tmp_prod1) \
               + np.dot(tmp_prod2, x_ta) \
               + np.dot(tmp_prod2, np.dot(B[a], tmp_prod1))
        p_ta = np.inner(z_ta, beta) \
               + np.dot(x_ta.T, theta[a]) \
               + ALPHA * np.sqrt(s_ta)

        x_ta_dct[a] = x_ta
        p_ta_dct[a] = p_ta
        z_ta_dct[a] = z_ta

    to_predict = max(articles, key=lambda k:p_ta_dct[k])
    prev_pred = (timestamp, to_predict, x_ta_dct[to_predict], z_ta_dct[to_predict])
    return to_predict
