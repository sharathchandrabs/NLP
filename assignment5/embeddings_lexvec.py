
import gensim
from glove import Glove
from nltk import word_tokenize
from glove import Corpus
import gensim.models.keyedvectors as word2vec
import math
import numpy as np

model = open('word-test.v1.txt','r')


def square_rooted(x):
	return math.sqrt(sum([a * a for a in x]))

def cosSimilarity(d, c_a, c_b, c_c,):
	x = d
	y = c_b - c_a + c_c
	numerator = sum(a * b for a, b in zip(x, y))
	denominator = square_rooted(x) * square_rooted(y)
	return (numerator / denominator)

	


lines = []
final_sentences = []
for sents in model:
	#print(sents)
	words = word_tokenize(sents)
	final_sentences.append(words)


#Word2Vec



#w2c = gensim.models.Word2Vec(final_sentences, min_count = 1)

wordVec = word2vec.KeyedVectors.load_word2vec_format("lexvec.commoncrawl.300d.W.pos.neg3.txt", binary=False)

#print(wordVec.similarity('baghdad', 'iraq'))






#GLOVE


# corpus = Corpus()
# corpus.fit("GoogleNews-vectors-negative300.bin", window=10)
# glove = Glove(no_components=100, learning_rate=0.05)
# glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
# glove.add_dictionary(corpus.dictionary)
# glove.most_similar('Iran', number=10)

# print(glove.most_similar('Iran', number=10))

#compare
model = open('word-test.v1.txt','r')
capital_word_sents = []
currency_sents = []
city_in_state_sents = []
family_sents = []
gram1_adjective_to_adverb_sents = []
status_gram2_opposite_sents = []
gram3_comparative_sents = []
gram6_nationality_adjective_sents = []

status_capital_world = False
status_currency = False
status_city_in_state = False
status_family = False
status_gram1_adjective_to_adverb = False
status_gram2_opposite = False
status_gram3_comparative = False
status_gram6_nationality_adjective = False


capital_word_sents_set = []
currency_sents_set = []
city_in_state_sents_set = []
family_sents_set = []
gram1_adjective_to_adverb_sents_set = []
status_gram2_opposite_sents_set = []
gram3_comparative_sents_set = []
gram6_nationality_adjective_sents_set = []

for sent in model:

	#======capital-word==========
	if ": capital-world" in sent:
		status_capital_world = True
		status_currency = False
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False

	if(status_capital_world):
		capital_word_sents_set+=(sent.split())
		capital_word_sents.append(sent)
	#======capital-word-ends=====

	#======currency==========
	if ": currency" in sent:
		status_capital_world = False
		status_currency = True
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False

	if(status_currency):
		currency_sents_set+=(sent.split())
		currency_sents.append(sent)
	#======currency-ends=====

	#======city-in-state,==========
	if ": city-in-state" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = True
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False

	if(status_city_in_state):
		city_in_state_sents_set+=(sent.split())
		city_in_state_sents.append(sent)
	#======city-in-state,-ends=====

	#======family==========
	if ": family" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = False
		status_family = True
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False

	if(status_family):
		family_sents_set+=(sent.split())
		family_sents.append(sent)
	#======family-ends=====

	#======gram1-adjective-to-adverb==========
	if ": gram1-adjective-to-adverb" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = True
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False

	if(status_gram1_adjective_to_adverb):
		gram1_adjective_to_adverb_sents_set+=(sent.split())
		gram1_adjective_to_adverb_sents.append(sent)
	#======gram1-adjective-to-adverb-ends=====

	#======gram2-opposite==========
	if ": gram2-opposite" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = True
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False

	if(status_gram2_opposite):
		status_gram2_opposite_sents_set+=(sent.split())
		status_gram2_opposite_sents.append(sent)
	#======gram2-opposite-ends=====

	#======gram3-comparative==========
	if ": gram3-comparative" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = True
		status_gram6_nationality_adjective = False

	if(status_gram3_comparative):
		gram3_comparative_sents_set+=(sent.split())
		gram3_comparative_sents.append(sent)
	#======gram3-comparative-ends=====
	if ": gram4-superlative" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False
		continue

	if ": gram7-past-tense" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False
		continue

	if ": gram8-plural" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False
		continue
	if ": gram9-plural-verbs" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = False
		continue

	#======gram6-nationality-adjective==========
	if ": gram6-nationality-adjective" in sent:
		status_capital_world = False
		status_currency = False
		status_city_in_state = False
		status_family = False
		status_gram1_adjective_to_adverb = False
		status_gram2_opposite = False
		status_gram3_comparative = False
		status_gram6_nationality_adjective = True

	if(status_gram6_nationality_adjective):
		gram6_nationality_adjective_sents_set+=(sent.split())
		gram6_nationality_adjective_sents.append(sent)
	#======gram6-nationality-adjective-ends=====


del capital_word_sents[0]
del capital_word_sents[len(capital_word_sents) - 1]

del currency_sents[0]
del currency_sents[len(currency_sents) - 1]

del city_in_state_sents[0]
del city_in_state_sents[len(city_in_state_sents) - 1]

del family_sents[0]
del family_sents[len(family_sents) - 1]

del gram1_adjective_to_adverb_sents[0]
del gram1_adjective_to_adverb_sents[len(gram1_adjective_to_adverb_sents) - 1]

del status_gram2_opposite_sents[0]
del status_gram2_opposite_sents[len(status_gram2_opposite_sents) - 1]



# print(gram3_comparative_sents)

# print("#"*1000)
del gram3_comparative_sents[0]
del gram3_comparative_sents[len(gram3_comparative_sents) - 1]
# print(gram3_comparative_sents)

del gram6_nationality_adjective_sents[0]
del gram6_nationality_adjective_sents[len(gram6_nationality_adjective_sents) - 1]
#print(gram6_nationality_adjective_sents)



del capital_word_sents_set[0]
del capital_word_sents_set[len(capital_word_sents_set) - 1]

del currency_sents_set[0]
del currency_sents_set[len(currency_sents_set) - 1]

del city_in_state_sents_set[0]
del city_in_state_sents_set[len(city_in_state_sents_set) - 1]

del family_sents_set[0]
del family_sents_set[len(family_sents_set) - 1]

del gram1_adjective_to_adverb_sents_set[0]
del gram1_adjective_to_adverb_sents_set[len(gram1_adjective_to_adverb_sents_set) - 1]

del status_gram2_opposite_sents_set[0]
del status_gram2_opposite_sents_set[len(status_gram2_opposite_sents_set) - 1]

del gram3_comparative_sents_set[0]
del gram3_comparative_sents_set[len(gram3_comparative_sents_set) - 1]

del gram6_nationality_adjective_sents_set[0]
del gram6_nationality_adjective_sents_set[len(gram6_nationality_adjective_sents_set) - 1]


capital_word_sents_set_final = set(capital_word_sents_set)
currency_sents_set_final = set(currency_sents_set)
city_in_state_sents_set_final = set(city_in_state_sents_set)
family_sents_set_final = set(family_sents_set)
gram1_adjective_to_adverb_sents_set_final = set(gram1_adjective_to_adverb_sents_set)
status_gram2_opposite_sents_set_final = set(status_gram2_opposite_sents_set)
gram3_comparative_sents_set_final = set(gram3_comparative_sents_set)
gram6_nationality_adjective_sents_set_final = set(gram6_nationality_adjective_sents_set)


#Word2Vec vectors
accuracy = 0
resultant_cosSim = dict()

for sentences in capital_word_sents:
	line = sentences.lower().split()
	for words in capital_word_sents_set_final:
		try:
			resultant_cosSim[words] = cosSimilarity(np.array(wordVec[words]), np.array(wordVec[line[0]]), np.array(wordVec[line[1]]), np.array(wordVec[line[2]]))
		except:
			continue
	#print(line[3], max(resultant_cosSim))
	try:
		if(line[3] == max(resultant_cosSim, key = resultant_cosSim.get)):
			accuracy+=1
	except:
		continue
#print("Accuracy for :capital-world class is",accuracy/len(capital_word_sents))






resultant_cosSim = dict()

for sentences in currency_sents:
	line = sentences.lower().split()
	for words in currency_sents_set_final:
		try:
			resultant_cosSim[words] = cosSimilarity(np.array(wordVec[words]), np.array(wordVec[line[0]]), np.array(wordVec[line[1]]), np.array(wordVec[line[2]]))
		except:
			continue
	#print(line[3], max(resultant_cosSim))
	try:
		if(line[3] ==max(resultant_cosSim, key = resultant_cosSim.get)):
			accuracy+=1
	except:
		continue
#print("Accuracy for :currency class is",accuracy/len(currency_sents))





resultant_cosSim = dict()


for sentences in city_in_state_sents:
	line = sentences.lower().split()
	for words in city_in_state_sents_set_final:
		try:
			resultant_cosSim[words] = cosSimilarity(np.array(wordVec[words]), np.array(wordVec[line[0]]), np.array(wordVec[line[1]]), np.array(wordVec[line[2]]))
		except:
			continue
	#print(line[3], max(resultant_cosSim))
	try:
		if(line[3] == max(resultant_cosSim, key = resultant_cosSim.get)):
			accuracy+=1
	except:
		continue

#print("Accuracy for :city-in-state class is",accuracy/len(city_in_state_sents))




resultant_cosSim = dict()


for sentences in family_sents:
	line = sentences.lower().split()
	for words in family_sents_set_final:
		try:
			resultant_cosSim[words] = cosSimilarity(np.array(wordVec[words]), np.array(wordVec[line[0]]), np.array(wordVec[line[1]]), np.array(wordVec[line[2]]))
		except:
			continue
	#print(line[3], max(resultant_cosSim))
	try:
		if(line[3] ==max(resultant_cosSim, key = resultant_cosSim.get)):
			accuracy+=1
	except:
		continue

#print("Accuracy for :family-sents is",accuracy/len(family_sents))




resultant_cosSim = dict()



for sentences in gram1_adjective_to_adverb_sents:
	line = sentences.lower().split()
	for words in gram1_adjective_to_adverb_sents_set_final:
		try:
			resultant_cosSim[words] = cosSimilarity(np.array(wordVec[words]), np.array(wordVec[line[0]]), np.array(wordVec[line[1]]), np.array(wordVec[line[2]]))
		except:
			continue
	#print(line[3], max(resultant_cosSim))
	try:
		if(line[3] == max(resultant_cosSim, key = resultant_cosSim.get)):
			accuracy+=1
	except:
		continue

#print("Accuracy for gram1-adjective-to-adverb class is",accuracy/len(gram1_adjective_to_adverb_sents))




resultant_cosSim = dict()


for sentences in status_gram2_opposite_sents:
	line = sentences.lower().split()
	for words in status_gram2_opposite_sents_set_final:
		try:
			resultant_cosSim[words] = cosSimilarity(np.array(wordVec[words]), np.array(wordVec[line[0]]), np.array(wordVec[line[1]]), np.array(wordVec[line[2]]))
		except:
			continue
	#print(line[3], max(resultant_cosSim))
	try:
		if(line[3] == max(resultant_cosSim, key = resultant_cosSim.get)):
			accuracy+=1
	except:
		continue
#print(resultant_cosSim)
#print("Accuracy for :gram2-opposite class is",accuracy/len(status_gram2_opposite_sents))



resultant_cosSim = dict()

# for s in gram3_comparative_sents:
# 	print(s)

for sentences in gram3_comparative_sents:
	line = sentences.lower().split()
	#print(line)
	for words in gram3_comparative_sents_set_final:
		try:
			resultant_cosSim[words] = cosSimilarity(np.array(wordVec[words]), np.array(wordVec[line[0]]), np.array(wordVec[line[1]]), np.array(wordVec[line[2]]))
		except:
			continue
	#print(line[3], max(resultant_cosSim))
	#print(line)
	try:
		if(line[3] == max(resultant_cosSim, key = resultant_cosSim.get)):
			accuracy+=1
	except:
		continue

#print("Accuracy for :gram3-comparative class is",accuracy/len(gram3_comparative_sents))





resultant_cosSim = dict()


for sentences in gram6_nationality_adjective_sents:
	line = sentences.lower().split()
	for words in gram6_nationality_adjective_sents_set_final:
		try:
			resultant_cosSim[words] = cosSimilarity(np.array(wordVec[words]), np.array(wordVec[line[0]]), np.array(wordVec[line[1]]), np.array(wordVec[line[2]]))
		except:
			continue
	#print(line[3], max(resultant_cosSim))
	try:
		if(line[3] == max(resultant_cosSim, key = resultant_cosSim.get)):
			accuracy+=1
	except:
		continue


final_count = len(gram6_nationality_adjective_sents)

print("Accuracy is",(accuracy/final_count)*100)

