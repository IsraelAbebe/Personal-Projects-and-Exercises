import io, sys, math, re
from collections import defaultdict
import numpy as np

# GOAL: build a stupid backoff bigram model

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

# Function to build a bigram model

def build_bigram(data):
    unigram_counts = defaultdict(lambda:0)
    bigram_counts  = defaultdict(lambda:defaultdict(lambda: 0.0))
    total_number_words = 0

    
    for sentence in data:
        for word in sentence:
            unigram_counts[word] += 1.0
            
    for s in range(len(data)):
        for w in range(len(data[s])-1):
            w1,w2 = data[s][w:w+2][0],data[s][w:w+2][1]
            bigram_counts[w1][w2] += 1.0
            
            
            
    # Store the unigram and bigram counts as well as the total 
    # number of words in the dataset

    unigram_prob = defaultdict(lambda:0)
    bigram_prob = defaultdict(lambda: defaultdict(lambda: 0.0))

    # Build unigram and bigram probabilities from counts
    for word in unigram_counts:
        unigram_prob[word] = unigram_counts[word]/sum(list(unigram_counts.values()))

    for w1 in bigram_counts:
        for w2 in bigram_counts[w1]:
            bigram_prob[w1][w2] = bigram_counts[w1][w2] / unigram_counts[w1]
        
    return {'bigram': bigram_prob, 'unigram': unigram_prob}

def get_prob(model, w1, w2):
    assert model["unigram"][w2] != 0, "Out of Vocabulary word!"
    
    if model["bigram"][w1][w2] != 0:
        return model["bigram"][w1][w2]
    else:
        return model["unigram"][w1]*0.4
    ## FILL CODE
    # Should return the probability of the bigram (w1w2) if it exists
    # Else it return the probility of unigram (w2) multiply by 0.4

def perplexity(model, data):
    perp = 0.0
    all_words = 0.0
    
    for s in range(len(data)):
        for w in range(len(data[s])-1):
            w1,w2 = data[s][w],data[s][w+1]
            perp += np.log(get_prob(model, w1, w2))
            all_words += 1.0
            
    # follow the formula in the slides
    # call the function get_prob to get P(w2 | w1)
    return np.exp(-perp/all_words)

def generate(model):
    sentence = ["<s>"]
    
    # CHOOSE MOST PROBABLE WORDS WITH STARTING SENTENCE
    letter = sentence[-1]
    while letter != '</s>':
        possible_words = list(model['bigram'][letter].keys())
        possible_prob = list(model['bigram'][letter].values())

        # # possible_prob
        word = np.random.choice(possible_words, 1,p=possible_prob)[0]
        sentence.append(word)
        letter = word
        
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    return sentence

###### MAIN #######

print("load training set")
train_data, vocab = load_data("train2.txt")
train_data = remove_rare_words(train_data, vocab,5)
# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# rare words with <unk> in the dataset

print("build bigram model")
model = build_bigram(train_data)

print("load validation set")
valid_data, _ = load_data("valid2.txt")
valid_data = remove_rare_words(valid_data, vocab,5)
# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# OOV with <unk> in the dataset

print("The perplexity is", perplexity(model, valid_data))

print("Generated sentence: ",generate(model))
