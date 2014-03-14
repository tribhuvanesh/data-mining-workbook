#!/usr/bin/env python

import numpy as np
import sys

# Constants to construct minhash signature
NUM_SIG_HASHES = 250
MOD_N = 1000
# Constants to partition the signature matrix
# Value of r
NUM_ROWS_PER_BAND = 25
# Value of b = NUM_SIG_HASHES/r


def partition(video_id, shingles, hash_coefficients):
    sig_mat = np.empty(NUM_SIG_HASHES)
    sig_mat[:] = np.inf
    # Construct a column of the signature matrix
    for i in xrange(1, 10000):
        if i in shingles:
            for h in xrange(len(hash_coefficients)):
                sig_mat[h] = min(sig_mat[h],
                                 (hash_coefficients[h][0] * i + hash_coefficients[h][1]) % hash_coefficients[h][2])

    # Partition the column into bands and print them as output of the mapper
    for i in range(0, NUM_SIG_HASHES, NUM_ROWS_PER_BAND):
        if i == 0:
            band_id = 0
        else:
            band_id = i/NUM_ROWS_PER_BAND
        # print sig_mat[1:10]
        # print ','.join([str(int(x)) for x in sig_mat[i:i + NUM_ROWS_PER_BAND]])
        print '%d:%s\t%s' % (band_id, ','.join([str(int(x)) for x in sig_mat[i:i + NUM_ROWS_PER_BAND]]), video_id)


if __name__ == "__main__":
    # Very important. Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)

    # Generate NUM_SIG_HASHES hash functions of the form ax + b mod n
    # [a_i b_i n]
    hash_coefficients = np.zeros((NUM_SIG_HASHES, 3))
    for h in xrange(NUM_SIG_HASHES):
        hash_coefficients[h][0] = np.random.randint(1, MOD_N)
        hash_coefficients[h][1] = np.random.randint(1, MOD_N)
        hash_coefficients[h][2] = MOD_N

    for line in sys.stdin:
        line = line.strip()
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], sep=" ")
        partition(video_id, shingles, hash_coefficients)