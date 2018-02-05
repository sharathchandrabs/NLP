import nltk
import math
import operator
import json

train_file = open("./data/train")
training_file_data = train_file.readlines()
docContentDict = dict()
speakerDict = dict()
priorProbs = dict()
speakerWordCount = dict()
vocab = []

speakerBagOfWords = dict()
speakerBagOfWordsCount = dict()
speakerWords = dict()
speakerWordsProbability = dict()
speakerWordsProbabilityUnique = dict()
bigramSpeakerWordsDict = dict()
speakerWordsBigramProbability = dict()
trigramSpeakerWordsDict = dict()
speakerWordsTrigramProbability = dict()
frequentlyUsedSpeakerWords = dict()
speakerWordsUnique = dict()
count = 0
print("Loading...")
for line in training_file_data:
	line_list = list(line.split())
	speaker = line_list[0]
	if speaker in speakerWords:
		speakerWords[speaker] += line_list[1:]
	else:
		speakerWords[speaker] = line_list[1:]

	if speaker in bigramSpeakerWordsDict:
		bigramSpeakerWordsDict[speaker] += list(nltk.ngrams(line_list[1:] , 2))
	else:
		bigramSpeakerWordsDict[speaker] = list(nltk.ngrams(line_list[1:] , 2))

	if speaker in trigramSpeakerWordsDict:
		trigramSpeakerWordsDict[speaker] += list(nltk.ngrams(line_list[1:] , 3))
	else:
		trigramSpeakerWordsDict[speaker] = list(nltk.ngrams(line_list[1:] , 3))


	if speaker in speakerWordsUnique:
		speakerWordsUnique[speaker] += list(set(line_list[1:]))
	else:
		speakerWordsUnique[speaker] = list(set(line_list[1:]))
	vocab+=line_list[1:]

	docContentDict[count] = line_list
	if speaker in speakerDict:
		speakerDict[speaker]+=1
	else:
		speakerDict[speaker] = 1
	count+=1

vocab = set(vocab)


#print(speakerWords['chafee'])
# print(bigramSpeakerWordsDict['chafee'])


#length = 0
for k,v in speakerWords.items():
	#prepare dictionary of dictionary where keys of first dictionary is all unique words used by speaker and their respective values will be their count of number of times used by the speaker.
	speakerWordsProbability[k] = dict(nltk.FreqDist(v))

for k,v in speakerWordsUnique.items():
	speakerWordsProbabilityUnique[k] = dict(nltk.FreqDist(v))
	


# print(speakerWordsProbability['sanders']['and'], speakerWordsProbabilityUnique['sanders']['and'])


# print(frequentlyUsedSpeakerWords)

vocabCount = len(vocab)



for k,v in speakerDict.items():
	priorProbs[k] = math.log(v/count)

#print(priorProbs)


for line in training_file_data:
	line_list = list(line.split())
	line_length = len(line_list) - 1
	speaker = line_list[0]
	if speaker in speakerWordCount:
		speakerWordCount[speaker]+=line_length
	else:
		speakerWordCount[speaker] = line_length

#applying likelihood estimation

likelihoodDict = dict()
likelihoodDictUnique = dict()
alpha = 0.15
for k,v in speakerWordsProbability.items():
	likelihoodDict[k] = {}
	for word in vocab:
		likelihoodDict[k][word] = math.log((v.get(word,0)+alpha)/(vocabCount*alpha + sum(v.values())))
		#likelihoodDictUnique[k][word] = 

#print(likelihoodDict['chafee']['and'])
for k,v in likelihoodDict.items():
	frequentlyUsedSpeakerWords[k] = dict(sorted(likelihoodDict[k].items(), key=operator.itemgetter(1), reverse=True)[:20])

#print(frequentlyUsedSpeakerWords)

#applying naive bayes

test_file = open("./data/test")
testing_file_data = test_file.readlines()


accuracy = 0
totalTestDocs = 0

for line in testing_file_data:
	line_split = list(line.split())
	speakerName = line_split[0]
	totalTestDocs+=1
	speakerText = line_split[1:]
	probabilityOfSpeaker = dict()
	for k,v in speakerWordsProbability.items():
		probabilityOfSpeaker[k] = sum([likelihoodDict.get(k).get(words, 0) or 0 for words in speakerText]) + priorProbs[k]


	
	predictedSpeaker = max(probabilityOfSpeaker.items(), key=operator.itemgetter(1))[0]

	if(predictedSpeaker == speakerName):
		accuracy+=1
#print(probabilityOfSpeaker['trump'])
print("Naive Bayes Model",(accuracy*100)/totalTestDocs)






#bigrams


for k,v in bigramSpeakerWordsDict.items():
	speakerWordsBigramProbability[k] = dict(nltk.FreqDist(v))

#print(speakerWordsBigramProbability['chafee'])

#apply likelihood
likelihoodBigramEstimate = dict()
alpha = 0.15
for k,v in speakerWordsBigramProbability.items():
	likelihoodBigramEstimate[k] = {}
	for bigrams in v:
		if (speakerWordsBigramProbability.get(k).get(bigrams) == None):
			likelihoodBigramEstimate[k][bigrams] = math.log((v.get(bigrams,0)+alpha)) - math.log(vocabCount*alpha + len(speakerWordsBigramProbability.get(k)))
		else:
			likelihoodBigramEstimate[k][bigrams] = math.log((v.get(bigrams,0)+alpha)) - math.log(vocabCount*alpha + speakerWordsBigramProbability.get(k).get(bigrams,len(speakerWordsBigramProbability.get(k))))


#applying naive bayes to bigrams model

accuracy = 0
totalTestDocs = 0

for line in testing_file_data:
	line_split = list(line.split())
	speakerName = line_split[0]
	totalTestDocs+=1
	speakerText = line_split[1:]
	probabilityOfSpeaker = dict()
	for k,v in speakerWordsBigramProbability.items():
		probabilityOfSpeaker[k] = sum([likelihoodBigramEstimate.get(k).get(speakerText[words-1]+","+speakerText[words], likelihoodDict.get(k).get(speakerText[words])) or 1 for words in range(len(speakerText)-1)]) + priorProbs[k]

	
	predictedSpeaker = max(probabilityOfSpeaker.items(), key=operator.itemgetter(1))[0]

	if(predictedSpeaker == speakerName):
		accuracy+=1
#print(probabilityOfSpeaker['trump'])
print("Bigram model accuracy",(accuracy*100)/totalTestDocs)










#Unique words model

#applying likelihood estimation

alpha = 0.15
for k,v in speakerWordsProbabilityUnique.items():
	likelihoodDictUnique[k] = {}
	for word in vocab:
		likelihoodDictUnique[k][word] = math.log((v.get(word,0)+alpha)/(vocabCount*alpha + sum(v.values())))
		#likelihoodDictUnique[k][word] = 

#print(likelihoodDictUnique['chafee']['and'])
#applying naive bayes unique words model

test_file = open("./data/test")
testing_file_data = test_file.readlines()


accuracy = 0
totalTestDocs = 0

for line in testing_file_data:
	line_split = list(line.split())
	speakerName = line_split[0]
	totalTestDocs+=1
	speakerText = line_split[1:]
	probabilityOfSpeaker = dict()
	for k,v in speakerWordsProbabilityUnique.items():
		probabilityOfSpeaker[k] = sum([likelihoodDictUnique.get(k).get(words, 0) or 0 for words in speakerText]) + priorProbs[k]


	
	predictedSpeaker = max(probabilityOfSpeaker.items(), key=operator.itemgetter(1))[0]

	if(predictedSpeaker == speakerName):
		accuracy+=1
#print(probabilityOfSpeaker['trump'])
print("Binarised Naive Bayes Model",(accuracy*100)/totalTestDocs)








#trigrams


# for k,v in trigramSpeakerWordsDict.items():
# 	speakerWordsTrigramProbability[k] = dict(nltk.FreqDist(v))

# #apply likelihood
# likelihoodTrigramEstimate = dict()
# alpha = 0.15
# for k,v in speakerWordsTrigramProbability.items():
# 	likelihoodTrigramEstimate[k] = {}
# 	for trigrams in v:
# 		bigram = (trigrams[0], trigrams[1])
# 		if (speakerWordsBigramProbability.get(k).get(bigram) == None):
# 			likelihoodTrigramEstimate[k][trigrams] = math.log((v.get(trigrams,0)+alpha)) - math.log(vocabCount*alpha + len(speakerWordsBigramProbability.get(k)))
# 		else:
# 			likelihoodTrigramEstimate[k][trigrams] = math.log((v.get(trigrams,0)+alpha)) - math.log(vocabCount*alpha + speakerWordsBigramProbability.get(k).get(trigrams[0],len(speakerWordsBigramProbability.get(k))))


#applying naive bayes to trigrams model

#print(likelihoodTrigramEstimate)

# accuracy = 0
# totalTestDocs = 0

# for line in testing_file_data:
# 	line_split = list(line.split())
# 	speakerName = line_split[0]
# 	totalTestDocs+=1
# 	speakerText = line_split[1:]
# 	probabilityOfSpeaker = dict()
# 	for k,v in speakerWordsTrigramProbability.items():

# 		probabilityOfSpeaker[k] = sum([likelihoodTrigramEstimate.get(k).get(speakerText[words-2]+","+speakerText[words-1]+","+speakerText[words], likelihoodBigramEstimate.get(k).get(speakerText[words]+","+speakerText[words-1])) or 1  for words in range(len(speakerText)-1)]) + priorProbs[k]

# 	# for k,v in speakerWordsTrigramProbability.items():
# 	# 	for words in range(len(speakerText)-1):
# 	# 		#print(speakerText[words-2]+","+speakerText[words-1]+","+speakerText[words],likelihoodTrigramEstimate.get(k).get(speakerText[words-2]+","+speakerText[words-1]+","+speakerText[words]), likelihoodBigramEstimate.get(k).get(speakerText[words-1]+","+speakerText[words]))

# 	predictedSpeaker = max(probabilityOfSpeaker.items(), key=operator.itemgetter(1))[0]

# 	if(predictedSpeaker == speakerName):
# 		accuracy+=1

# print("trigram accuracy",(accuracy*100)/totalTestDocs)


# path = "./RScript/output.txt"
# f = open(path, 'w')

# jsonOutput = json.dumps(frequentlyUsedSpeakerWords)
# f.write(jsonOutput)
# f.close()

# for k,v in frequentlyUsedSpeakerWords.items():
# 	for k2,v2 in v.items():
# 		print(k + "\t" +k2 + "\t" + str(v2))