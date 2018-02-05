import nltk
import string
from nltk.corpus import udhr  
from collections import Counter
from nltk import FreqDist
import math


english = udhr.raw('English-Latin1') 
english_train, english_dev = english[0:1000], english[1000:1100] 
english_test = udhr.words('English-Latin1')[0:1000] 


french = udhr.raw('French_Francais-Latin1') 
french_train, french_dev = french[0:1000], french[1000:1100] 
french_test = udhr.words('French_Francais-Latin1')[0:1000]


spanish = udhr.raw('Spanish_Espanol-Latin1')
spanish_train, spanish_dev = spanish[0:1000], spanish[1000:1100] 
spanish_test = udhr.words('Spanish_Espanol-Latin1')[0:1000]


italian = udhr.raw('Italian_Italiano-Latin1') 
italian_train, italian_dev = italian[0:1000], italian[1000:1100] 
italian_test = udhr.words('Italian_Italiano-Latin1')[0:1000] 


#preprocessing input training data
def preProcessAndCount(train):
	trainSet = []
	for chars in train:
		if chars.isalnum():
			trainSet.append(chars.lower())
		elif chars == ' ':
			trainSet.append(chars.lower())
	#print(trainSet)
	freqDist = nltk.FreqDist(trainSet)
	return trainSet,freqDist

#calculating probabilities of unigrams
def unigramCount(charDict):
	unigram_count={}
	for word in charDict:
		unigram_count[word]=(charDict[word]/sum(charDict.values()))
	return unigram_count


#calculating probabilities of bigrams                
def bigramCount(bigramArray,charDict):
	bigram_count_dictionary={}
	freqd=nltk.FreqDist(bigramArray)
	for w1,w2 in freqd.keys():
		for word in charDict.keys():
			if word==w1:
				bigram_count_dictionary[w1,w2]=(freqd[w1,w2]/charDict.get(word))
	return bigram_count_dictionary,freqd
                    
#calculating probabilities of triigrams
def trigramCount(trigramArray,charDict):
    trigram_count_dictionary={}
    freqd=nltk.FreqDist(trigramArray)
    for w1,w2,w3 in freqd.keys():
            for w4,w5 in charDict.keys():
                if w1==w4 and w2==w5: 
                    trigram_count_dictionary[w1,w2,w3]=(freqd[w1,w2,w3])/(charDict[w4,w5])
    return trigram_count_dictionary

#unigrams english
english_train_set, character_count_english = preProcessAndCount(english_train)
english_unigrams=list(nltk.ngrams(english_train_set,1))
english_unigram_count=unigramCount(character_count_english)
#bigrams english
english_bigrams = list(nltk.ngrams(english_train_set,2))
english_bigram_count,english_bigram_freqDist=bigramCount(english_bigrams,character_count_english)
#trigrams english
english_trigrams=list(nltk.ngrams(english_train_set,3))
english_trigram_count=trigramCount(english_trigrams,english_bigram_count)


#unigrams french
french_train_set, character_count_french = preProcessAndCount(french_train)
french_unigrams=list(nltk.ngrams(french_train_set,1))
french_unigram_count=unigramCount(character_count_french)
#bigrams french
french_bigrams = list(nltk.ngrams(french_train_set,2))
french_bigram_count,french_bigram_freqDist=bigramCount(french_bigrams,character_count_french)
#trigrams french
french_trigrams=list(nltk.ngrams(french_train_set,3))
french_trigram_count=trigramCount(french_trigrams,french_bigram_count)



#unigrams spanish
spanish_train_set, character_count_spanish = preProcessAndCount(spanish_train)
spanish_unigrams=list(nltk.ngrams(spanish_train_set,1))
spanish_unigram_count=unigramCount(character_count_spanish)
#biigrams spanish
spanish_bigrams = list(nltk.ngrams(spanish_train_set,2))
spanish_bigram_count,spanish_bigram_freqDist=bigramCount(spanish_bigrams,character_count_spanish)
#triigrams spanish
spanish_trigrams=list(nltk.ngrams(spanish_train_set,3))
spanish_trigram_count=trigramCount(spanish_trigrams,spanish_bigram_count)



#unigrams italian
italian_train_set, character_count_italian = preProcessAndCount(italian_train)
italian_unigrams=list(nltk.ngrams(italian_train_set,1))
italian_unigram_count=unigramCount(character_count_italian)
#bigrams italian
italian_bigrams = list(nltk.ngrams(italian_train_set,2))
italian_bigram_count,italian_bigram_freqDist=bigramCount(italian_bigrams,character_count_italian)
#triigrams italian
italian_trigrams=list(nltk.ngrams(italian_train_set,3))
italian_trigram_count=trigramCount(italian_trigrams,italian_bigram_count)




#==================================================================================================================================================


#ENGLISH VS FRENCH


#==================================================================================================================================================





def english_vs_french_unigram():
	english_accuracy = 0
	french_accuracy = 0
	for word in english_test:	#fetch every word in test set
		english_probs=[]
		french_probs=[]
		english=1
		french=1
		for c1 in word:
			for c2 in english_unigram_count.keys(): #every character in unigram dictionary
				if c1==c2:			#compare with character fetched
					english_probs.append(english_unigram_count[c1])
			for c3 in french_unigram_count.keys():
				if c1==c3:
					french_probs.append(french_unigram_count[c1])
		for values in french_probs:	#calculate probability by multiplying recursively
			french*=values
		for values in english_probs:
			english*=values
		if french<=english:
			english_accuracy+=1	#calculating accuracy based on number of hits
		if french>english:
			french_accuracy+=1
	return english_accuracy,french_accuracy

englishUnigramAccuracy,frenchUnigramAccuracy=english_vs_french_unigram()
print("Unigram accuracy of english model is ", (englishUnigramAccuracy/len(english_test))*100)
print("Unigram accuracy of french model is ", (frenchUnigramAccuracy/len(english_test))*100)


def english_vs_french_bigram():
    buf=''
    english_probs=[]
    french_probs=[]
    english=1
    french=1
    english_accuracy = 0
    french_accuracy = 0
    for word in english_test:
        if len(word)>1:
            buf=list(nltk.ngrams(word,2))
            for c1 in buf:
                for c2 in english_bigram_count.keys():
                    if c1==c2:
                        english_probs.append(english_bigram_count[c1])
                for c3 in french_bigram_count.keys():
                    if c1==c3:
                        french_probs.append(french_bigram_count[c1])
            for values in french_probs:
                french*=values
            for values in english_probs:
                english*=values
            if french<=english:
                english_accuracy+=1
            elif french>english:
                french_accuracy+=1
        english_probs=[]
        french_probs=[]
        english=1
        french=1
    return english_accuracy,french_accuracy


englishBigramAccuracy,frenchBigramAccuracy=english_vs_french_bigram()
print("Bigram accuracy of english model is ", (englishBigramAccuracy/len(english_test))*100)
print("Bigram accuracy of french model is ", (frenchBigramAccuracy/len(english_test))*100)


def english_vs_french_Trigram():
	buf=''
	english_probs=[]
	french_probs=[]
	english=1
	french=1
	english_accuracy = 0
	french_accuracy = 0
	for word in english_test:
		if len(word)>2:
			buf=list(nltk.ngrams(word,3))
			for c1 in buf:
				for c2 in english_trigram_count.keys():
					if c1==c2:
						english_probs.append(english_trigram_count[c1])
				for c3 in french_trigram_count.keys():
					if c1==c3:
						french_probs.append(french_trigram_count[c1])
			for values in french_probs:
				french+=values
			for values in english_probs:
				english+=values
			if french<=english:
				english_accuracy+=1
			elif french>english:
				french_accuracy+=1
			english_probs=[]
			french_probs=[]
			english=1
			french=1
	return english_accuracy,french_accuracy


englishTrigramAccuracy,frenchTrigramAccuracy=english_vs_french_Trigram()
print("Trigram accuracy of english model is ", (englishTrigramAccuracy/len(english_test))*100)
print("Trigram accuracy of french model is ", (frenchTrigramAccuracy/len(english_test))*100)






#==================================================================================================================================================


#SPANISH VS ITALIAN


#==================================================================================================================================================



def spanish_vs_italian_unigram():
	spanish_accuracy = 0
	italian_accuracy = 0
	for word in spanish_test:
		spanish_probs=[]
		italian_probs=[]
		spanish=1
		italian=1
		for c1 in word:
			for c2 in spanish_unigram_count.keys():
				if c1==c2:
					spanish_probs.append(spanish_unigram_count[c1])
			for c3 in italian_unigram_count.keys():
				if c1==c3:
					italian_probs.append(italian_unigram_count[c1])
		for values in italian_probs:
			italian*=values
		for values in spanish_probs:
			spanish*=values
		if italian<=spanish:
			spanish_accuracy+=1
		if italian>spanish:
			italian_accuracy+=1
	return spanish_accuracy,italian_accuracy

spanishUnigramAccuracy,italianUnigramAccuracy=spanish_vs_italian_unigram()
print("Unigram accuracy of spanish model is ", (spanishUnigramAccuracy/len(spanish_test))*100)
print("Unigram accuracy of italian model is ", (italianUnigramAccuracy/len(spanish_test))*100)


def spanish_vs_italian_bigram():
    buf=''
    spanish_probs=[]
    italian_probs=[]
    spanish=1
    italian=1
    spanish_accuracy = 0
    italian_accuracy = 0
    for word in spanish_test:
        if len(word)>1:
            buf=list(nltk.ngrams(word,2))
            for c1 in buf:
                for c2 in spanish_bigram_count.keys():
                    if c1==c2:
                        spanish_probs.append(spanish_bigram_count[c1])
                for c3 in italian_bigram_count.keys():
                    if c1==c3:
                        italian_probs.append(italian_bigram_count[c1])
            for values in italian_probs:
                italian*=values
            for values in spanish_probs:
                spanish*=values
            if italian<=spanish:
                spanish_accuracy+=1
            elif italian>spanish:
                italian_accuracy+=1
        spanish_probs=[]
        italian_probs=[]
        spanish=1
        italian=1
    return spanish_accuracy,italian_accuracy


spanishBigramAccuracy,italianBigramAccuracy=spanish_vs_italian_bigram()
print("Bigram accuracy of spanish model is ", (spanishBigramAccuracy/len(spanish_test))*100)
print("Bigram accuracy of italian model is ", (italianBigramAccuracy/len(spanish_test))*100)


def spanish_vs_italian_Trigram():
	buf=''
	spanish_probs=[]
	italian_probs=[]
	spanish=1
	italian=1
	spanish_accuracy = 0
	italian_accuracy = 0
	for word in spanish_test:
		if len(word)>2:
			buf=list(nltk.ngrams(word,3))
			for c1 in buf:
				for c2 in spanish_trigram_count.keys():
					if c1==c2:
						spanish_probs.append(spanish_trigram_count[c1])
				for c3 in italian_trigram_count.keys():
					if c1==c3:
						italian_probs.append(italian_trigram_count[c1])
			for values in italian_probs:
				italian+=values
			for values in spanish_probs:
				spanish+=values
			if italian<=spanish:
				spanish_accuracy+=1
			elif italian>spanish:
				italian_accuracy+=1
			spanish_probs=[]
			italian_probs=[]
			spanish=1
			italian=1
	return spanish_accuracy,italian_accuracy


spanishTrigramAccuracy,italianTrigramAccuracy=spanish_vs_italian_Trigram()
print("Trigram accuracy of spanish model is ", (spanishTrigramAccuracy/len(spanish_test))*100)
print("Trigram accuracy of italian model is ", (italianTrigramAccuracy/len(spanish_test))*100)


