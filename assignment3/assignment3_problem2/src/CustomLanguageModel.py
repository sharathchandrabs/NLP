import re, collections,math, nltk
class CustomLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.laplaceBigramCounts = collections.defaultdict(lambda: 0)
    self.laplaceUnigramCounts = collections.defaultdict(lambda: 0)
    self.laplaceTrigramCounts = collections.defaultdict(lambda: 0)
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
        self.total+=1


    for sentence in corpus.corpus:
      for datum in sentence.data:
        token = datum.word
        self.entireText.append(token)

    token_bigram = list(nltk.ngrams(self.entireText, 2))
    token_trigram = list(nltk.ngrams(self.entireText, 3))
    #print(token_bigram)
    for tokens in token_bigram:
        self.laplaceBigramCounts[tokens] = self.laplaceBigramCounts[tokens] + 1
        
    
    for tokens in token_trigram:
        self.laplaceTrigramCounts[tokens] = self.laplaceTrigramCounts[tokens] + 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here

    score = 0.0
    token_tri = nltk.ngrams(sentence, 3)
    for tri in token_tri:
          bigramOfTri = list(nltk.ngrams(tri,2))
          count = self.laplaceTrigramCounts[tri]
          #print(tri, bigramOfTri[0] ,self.laplaceBigramCounts[bigramOfTri[0]])
          score += math.log(count+1)
          score -= math.log(len(self.laplaceBigramCounts) + self.laplaceBigramCounts[bigramOfTri[0]])
          #print(score)
      
    return score
