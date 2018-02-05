import re, collections,math, nltk
class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.laplaceBigramCounts = collections.defaultdict(lambda: 0)
    self.laplaceUnigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.entireText = []
    # TODO your code here
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here


    for sentence in corpus.corpus:
      for datum in sentence.data:
        token = datum.word
        self.laplaceUnigramCounts[token] = self.laplaceUnigramCounts[token] + 1


    for sentence in corpus.corpus:
      for datum in sentence.data:
        token = datum.word
        self.entireText.append(token)

    token_bigram = list(nltk.ngrams(self.entireText, 2))
    #print(token_bigram)
    for tokens in token_bigram:
        #print(tokens)
        self.laplaceBigramCounts[tokens] = self.laplaceBigramCounts[tokens] + 1
        self.total+=1



  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here

    score = 0.0
    token_bigram = nltk.ngrams(sentence, 2)
    for bi in token_bigram:
          count = self.laplaceBigramCounts[bi]
          score += math.log(count+1)
          score -= math.log(len(self.laplaceBigramCounts) + self.laplaceUnigramCounts[bi[0]])
          #print(score)
      
    return score
