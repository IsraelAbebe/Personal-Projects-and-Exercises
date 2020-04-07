import io, sys
import numpy as np
from heapq import *

def load_vectors(filename):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
    return data

## This function computes the cosine similarity between vectors u and v

def cosine(u, v):
    return np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

## This function returns the word corresponding to 
## nearest neighbor vector of x
## The list exclude_words can be used to exclude some
## words from the nearest neighbors search

def nearest_neighbor(x, word_vectors, exclude_words=[]):
    best_score = -1.0
    best_word = ''
    
    for k in word_vectors.keys():
        score = cosine(x,np.array(word_vectors[k]))
        
        if score > best_score and k not in exclude_words :
            best_score = score
            best_word = k

    return best_word

## This function return the words corresponding to the
## K nearest neighbors of vector x.
## You can use the functions heappush and heappop.

def knn(x, vectors, k):
    heap = []
    
    exclude_words=[]
    for i  in range(k+1):
        best_word = nearest_neighbor(x, word_vectors,exclude_words)
        exclude_words.append(best_word)
        heap.append((cosine(x,word_vectors[best_word]),best_word))
        
    return heap[1:]

## This function return the word d, such that a:b and c:d
## verifies the same relation

def analogy(a, b, c, word_vectors):
    result_word = None
    result_score = float('-Inf')
    
    x_a = word_vectors[a]/np.linalg.norm(word_vectors[a])
    x_b = word_vectors[b]/np.linalg.norm(word_vectors[b])
    x_c = word_vectors[c]/np.linalg.norm(word_vectors[c])
    for key in word_vectors:
        if True in [i in key for i in [a,b,c]] :
            continue
        normalized = word_vectors[key]/np.linalg.norm(word_vectors[key])
        result = np.dot((x_c+x_b-x_a),normalized)

        if result > result_score:
            result_score = result
            result_word = key
    
    return result_word 

## Compute the association strength between:
##   - a word w
##   - two sets of attributes A and B

def association_strength(w, A, B, vectors):
    strength_a = 0.0
    strength_b = 0.0
    ## FILL CODE
    for a in A:
        strength_a += ((1/len(A))*cosine(vectors[w],vectors[a]))
    for b in B:
        strength_b += ((1/len(B))*cosine(vectors[w],vectors[b]))
    
    return strength_a - strength_b

## Perform the word embedding association test between:
##   - two sets of words X and Y
##   - two sets of attributes A and B

def weat(X, Y, A, B, vectors):
    score_x = 0.0
    score_y = 0.0
    ## FILL CODE
    for i in X:
        score_x += association_strength(i,A,B,vectors)
    for i in Y:
        score_y += association_strength(i,A,B,vectors)

    return score_x - score_y

######## MAIN ########

print('')
print(' ** Word vectors ** ')
print('')

word_vectors = load_vectors(sys.argv[1])

print('similarity(apple, apples) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['apples']))
print('similarity(apple, banana) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['banana']))
print('similarity(apple, tiger) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['tiger']))

print('')
print('The nearest neighbor of cat is: ' +
      nearest_neighbor(word_vectors['cat'], word_vectors))

knn_cat = knn(word_vectors['cat'], word_vectors, 5)
print('')
print('cat')
print('--------------')
for score, word in knn(word_vectors['cat'], word_vectors, 5):
    print( word + '\t%.3f' % score)

print('')
print('france - paris + rome = ' + analogy('paris', 'france', 'rome', word_vectors))

## A word about biases in word vectors:

print('')
print('similarity(genius, man) = %.3f' %
      cosine(word_vectors['man'], word_vectors['genius']))
print('similarity(genius, woman) = %.3f' %
      cosine(word_vectors['woman'], word_vectors['genius']))

## Replicate one of the experiments from:
##
## Semantics derived automatically from language corpora contain human-like biases
## Caliskan, Bryson, Narayanan (2017)

career = ['executive', 'management', 'professional', 'corporation', 
          'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family',
          'cousins', 'marriage', 'wedding', 'relatives']
male = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
female = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']

print('')
print('Word embedding association test: %.3f' %
      weat(career, family, male, female, word_vectors))
