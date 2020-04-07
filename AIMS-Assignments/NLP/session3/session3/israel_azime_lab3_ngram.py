import io, sys, math, re
from collections import defaultdict
import numpy as np

# GOAL: build a stupid backoff ngram model

def load_data(filename):
    fin = io.open(filename, 'r', encoding='utf-8')
    data = []
    vocab = defaultdict(lambda:0)
    for line in fin:
        sentence = line.split()
        data.append(sentence)
        for word in sentence:
            vocab[word] += 1
    return data, vocab

def remove_rare_words(data, vocab, mincount):
    for sent in range(len(data)):
        for word_index in range(len(data[sent])):
            if vocab[data[sent][word_index]] < mincount:
                data[sent][word_index] = '<unk>'
    # replace words in data that are not in the vocab 
    # or have a count that is below mincount
    return data


def build_ngram(data, n):
    total_number_words = 0
    counts = defaultdict(lambda: defaultdict(lambda: 0.0))
    counts_n1 = defaultdict(lambda: 0.0)

    for sentence in data:
        #sentence = tuple(sentence)
        for w in range(len(sentence)-1):
            wt = sentence[w:w+n][-1]
            new_n = n-1
            counts_n1[wt] += 1.0
            while new_n >= 0:
                if new_n == 0:
                    wt_n = '<empty>'
                else:
                    wt_n = ' '.join(sentence[w:w+new_n])
                    
                counts[wt_n][wt] += 1.0
                counts_n1[wt_n] += 1.0
                total_number_words += 1.0
                new_n -= 1
            
        # dict can be indexed by tuples
        # store in the same dict all the ngrams
        # by using the context as a key and the word as a value

    prob  = defaultdict(lambda: defaultdict(lambda: 0.0))
    for wt_n in counts:
        for wt in counts[wt_n]:
            prob[wt_n][wt] = counts[wt_n][wt] / sum(counts[wt_n].values())
    # Build the probabilities from the counts
    # Be careful with how you normalize!

    return prob

def get_prob(model, context, w):
    prob = 0.0
    # START FROM N-GRAM AND DO N-N GRAM UNTIL YOU REACH BIGRAM
    # IF YOU GET NON ZERO PROBAVLITY IN THE MIDDLE RETURN IT 
    new_context = context.split(' ')
    for i in range(len(new_context)):
        prob = model[' '.join(new_context[i:])][w]
        if prob != 0:
            return prob
        
        
    # if you dont have the bigram of the word just calculate the unigram and return   
    if model['<empty>'][w]*0.4 !=0:
        return model['<empty>'][w]*0.4
    else:
        return (0+1)/(len(model['<empty>'].keys())+sum(model['<empty>'].values()))
    
    # code a recursive function over 
    # smaller and smaller context
    # to compute the backoff model
    # Bonus: You can also code an interpolation model this way
def perplexity(model, data, n):
    perp = 0.0
    all_words = 0.0
    for sentence in data:
        for w in range(len(sentence)-1):
            w,context = sentence[w:w+n][-1],' '.join(sentence[w:w+n][:-1])
            prob = get_prob(model, context, w)
            perp += np.log(prob)
            all_words += 1.0
            
    # Same as bigram.py
    return np.exp(-perp/all_words)

def get_proba_distrib(model, context):
    if sum(model[context].values()) != 0:
        return context
    else:
        context = ' '.join(context.split(' ')[:-1])
        return get_proba_distrib(model,context)
    # code a recursive function over context
    # to find the longest available ngram 

def generate(model):
    sentence = ["<s>"]
    
    MAX_LEN = 50
    n = 0
    while sentence[-1] != '</s>':
        letter = get_proba_distrib(model,' '.join(sentence))
        possible_words = list(model[letter].keys())
        possible_prob = list(model[letter].values())

        # # possible_prob
        word = np.random.choice(possible_words, 1,p=possible_prob)[0]
        sentence.append(word)
        n+=1
        if n > MAX_LEN:
            break
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    return sentence

###### MAIN #######

n = 2

print("load training set")
train_data, vocab = load_data("train.txt")
train_data = remove_rare_words(train_data, vocab,5)
# Same as bigram.py

print("build ngram model with n = ", n)
model = build_ngram(train_data, n)

print("load validation set")
valid_data, _ = load_data("valid.txt")
valid_data = remove_rare_words(valid_data, vocab,5)
# Same as bigram.py

print("The perplexity is", perplexity(model, valid_data, n))

print("Generated sentence: ",generate(model))

