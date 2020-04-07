import io, sys
import numpy as np
from heapq import *

def load_vectors(filename):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))
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

    ## FILL CODE
    for word in word_vectors:
        if word not in exclude_words:
            #print(word)
            dist=cosine(x,word_vectors[word])
            if dist>best_score:
                best_score=dist
                best_word=word

    return best_word

## This function return the words corresponding to the
## K nearest neighbors of vector x.
## You can use the functions heappush and heappop.

def knn(x, vectors, k):
    heap = []
    for w in vectors:
        dist=cosine(x,vectors[w])
        heappush(heap, (dist,w))
        if len(heap)>k+1:
            heappop(heap)
            
     ## FILL CODE

    return [heappop(heap) for i in range(len(heap))][::-1][1:]

## This function return the word d, such that a:b and c:d
## verifies the same relation

def analogy(a, b, c, word_vectors):
    ## FILL CODE
    score=-1
    best=''
    
    x_a=word_vectors[a.lower()]
    x_b=word_vectors[b.lower()]
    x_c=word_vectors[c.lower()]
    x_a=x_a/np.linalg.norm(x_a)
    x_b=x_b/np.linalg.norm(x_b)
    x_c=x_c/np.linalg.norm(x_c)
    
    sim=x_c+x_b-x_a
    
    for word in word_vectors.keys():
        if True not in [i in word for i in [a,b,c]] :
            word_x=word_vectors[word]/np.linalg.norm(word_vectors[word])
            dist=np.dot(sim,word_x)
            
            if dist>score:
                score=dist
                best=word
    
    return best

## Compute the association strength between:
##   - a word w
##   - two sets of attributes A and B
def association_strength(w, A, B, vectors):
    strength_a = 0.0
    strength_b = 0.0
    ## FILL CODE
    for a in A:
        strength_a += cosine(vectors[w],vectors[a])
    for b in B:
        strength_b += cosine(vectors[w],vectors[b])
        
    return ((1/len(A))*strength_a)-((1/len(B))*strength_b)

## Perform the word embedding association test between:
##   - two sets of words X and Y
##   - two sets of attributes A and B

def weat(X, Y, A, B, vectors):
    score_x = 0.0
    score_y = 0.0
    ## FILL CODe
    for i in X:
        score_x +=association_strength(i,A,B,vectors)
    for j in Y:
        score_y +=association_strength(j,A,B,vectors)

    return score_x-score_y

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
    print (word + '\t%.3f' % score)

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
