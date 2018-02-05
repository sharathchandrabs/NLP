import nltk
import re
import string
import csv
import operator
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
from collections import Counter



#---------------Question 1----------------------
def split_text(text):
	words = text.split()
	print("-----------Question 1---------")
	for word in words:
		print(word)

def extract_second(text):
	words = text.split()
	w = ""
	for word in words:
		w+=word[1]
	print(w)

def list_phrases(text):
	words = text.split()
	phrases = []
	for word in words:
		phrases.append(word)
	index = phrases.index("sleep")
	print(phrases[:index])
	return phrases[:index]

def list_phrases_join(words_list):
	print (" ".join(words_list))

def alphabetical(text):
	words = text.split()
	phrases = []
	for word in words:
		phrases.append(word)
	phrases.sort()
	list_phrases_join(phrases)

funny = "colorless green ideas sleep furiously"
split_text(funny)
extract_second(funny)
phrases_list = list_phrases(funny)
list_phrases_join(phrases_list)
alphabetical(funny)
#-----------------------------------------------

#----------------Question 2---------------------
def wordfrequency(text):
	wordlist = []
	freqDict = {}
	words = text.split()
	print("\n\n-----------Question 2---------")
	for word in words:
		wordlist.append(words.count(word))
		freqDict[word] = words.count(word)
	print(freqDict)
sentence = input("Enter a Sentence:")
wordfrequency(sentence)
#-----------------------------------------------

#---------------Question 3----------------------
print("\n\n---------------Question 3----------------------")
print("3a. \\b(a|an|the)\\b")
print("3b. ([-+]?[0-9]*\.?[0-9]+[\/\+\-\*])+([-+]?[0-9]*\.?[0-9]+)")

#-----------------------------------------------

#---------------Question 4----------------------
import re

def parseLogin(text):
	print("\n\n-----------Question 4---------")
	ids = re.findall(r"[a-zA-Z0-9.]+@[a-zA-Z.]+[.][a-z]+", text)
	print(ids)

file_content = "austen-emma.txt:hart@vmd.cso.uiuc.edu (internet) hart@uiucvmd (bitnet)austen-emma.txt:Internet(72600.2026@compuserve.com); TEL: (212-254-5093)austen-persuasion.txt:Editing by Martin Ward (Martin.Ward@uk.ac.durham)blake-songs.txt:Prepared by David Price, email ccx074@coventry.ac.uk"

parseLogin(file_content)
#-----------------------------------------------

#---------------Question 5----------------------
print("\n\n-----------Question 5---------")
fileRead = input("Q5.Enter File Name(use existing input.txt if needed):")
infilename = fileRead
outfilename = "output.txt" 
lines_seen = set() # holds lines already seen
outfile = open(outfilename, "w")
for line in open(infilename, "r"):
    if line not in lines_seen: # not a duplicate
        outfile.write(line)
        lines_seen.add(line)
print("\n\nOpen generated outputfile.txt to check for duplicate lines removed\n")
outfile.close()

#-----------------------------------------------


#---------------Question 6----------------------
print("\n\n-----------Question 6---------")
print("The code iterates over the given input file one line after the other. Program will break when it encounters an empty line or a line which starts with # symbol. Further it replaces the symbol '(' with '(.' and ')' with '.)', after which it returns everythiong with brackets '()'.")
#-----------------------------------------------

#---------------Question 7----------------------
print("\n\n-----------Question 7---------")
inputFile = open("debate.txt", "r")
text = inputFile.read()
bracketRegex = '\(\w+\)'
	


sentences = nltk.sent_tokenize(text[293:-78])
#print("\n\n",sentences, "\n\n")


lehrer = []
obama = []
romney = []
prevSpkr = ''
for sentence in sentences:
	if re.search("\(\w+\)", sentence):
		continue
	#print(sentence)
	if re.search("^[LEHRER]+:", sentence):	
		prevSpkr = 'lehrer'
		lehrer.append(sentence[8:])
	elif re.search("^[OBAMA]+:", sentence):
		prevSpkr = 'obama'
		obama.append(sentence[7:])
	elif re.search("^[ROMNEY]+:", sentence):
		prevSpkr = 'romney'
		romney.append(sentence[8:])
	elif prevSpkr == 'lehrer':
		lehrer.append(sentence)
	elif prevSpkr == 'obama':
		obama.append(sentence)
	elif prevSpkr == 'romney':
		romney.append(sentence)



lehrer_tokenised = []
obama_tokenised = []
romney_tokenised = []

translator = str.maketrans('','',string.punctuation)

lehrer_tokenised_stopwords = []
obama_tokenised_stopwords = []
romney_tokenised_stopwords = []

for sents in lehrer:
	temp_sent = sents.translate(translator)
	temp_sent = temp_sent.lower()
	lehrer_tokenised.append(nltk.word_tokenize(temp_sent))
	

for sents in obama:
	temp_sent = sents.translate(translator)
	temp_sent = temp_sent.lower()
	obama_tokenised.append(nltk.word_tokenize(temp_sent)) 

for sents in romney:
	temp_sent = sents.translate(translator)
	temp_sent = temp_sent.lower()
	romney_tokenised.append(nltk.word_tokenize(temp_sent)) 

for stpwords in lehrer_tokenised:
	for w in stpwords:
		if w not in stopwords.words("english"):
			lehrer_tokenised_stopwords.append(w)

for stpwords in obama_tokenised:
	for w in stpwords:
		if w not in stopwords.words("english"):
			obama_tokenised_stopwords.append(w)

for stpwords in romney_tokenised:
	for w in stpwords:
		if w not in stopwords.words("english"):
			romney_tokenised_stopwords.append(w)

#print(lehrer_tokenised_stopwords)


#apply stemmers
#for Lehrer
pstemmer = PorterStemmer()
lehrer_tokenised_psingles = [pstemmer.stem(plural) for plural in lehrer_tokenised_stopwords]

sstemmer = SnowballStemmer("english")
lehrer_tokenised_ssingles = [sstemmer.stem(plural) for plural in lehrer_tokenised_stopwords]

lstemmer = LancasterStemmer()
lehrer_tokenised_lsingles = [sstemmer.stem(plural) for plural in lehrer_tokenised_stopwords]

#print (lehrer_tokenised_lsingles)

#for Obama
pstemmer = PorterStemmer()
obama_tokenised_psingles = [pstemmer.stem(plural) for plural in obama_tokenised_stopwords]

sstemmer = SnowballStemmer("english")
obama_tokenised_ssingles = [sstemmer.stem(plural) for plural in obama_tokenised_stopwords]

lstemmer = LancasterStemmer()
obama_tokenised_lsingles = [sstemmer.stem(plural) for plural in obama_tokenised_stopwords]

#print (obama_tokenised_lsingles)

#for Romney
pstemmer = PorterStemmer()
romney_tokenised_psingles = [pstemmer.stem(plural) for plural in romney_tokenised_stopwords]

sstemmer = SnowballStemmer("english")
romney_tokenised_ssingles = [sstemmer.stem(plural) for plural in romney_tokenised_stopwords]

lstemmer = LancasterStemmer()
romney_tokenised_lsingles = [sstemmer.stem(plural) for plural in romney_tokenised_stopwords]

#print (romney_tokenised_lsingles)

#top 10 lehrer
counts_pstem_lehrer = dict(Counter(lehrer_tokenised_psingles))
sorted_counts_pstem_lehrer = sorted(counts_pstem_lehrer.items(), key=operator.itemgetter(1))
print("Porter stemmer output for Lehrer:\t",sorted_counts_pstem_lehrer[-10:])

counts_sstem_lehrer = dict(Counter(lehrer_tokenised_ssingles))
sorted_counts_sstem_lehrer = sorted(counts_sstem_lehrer.items(), key=operator.itemgetter(1))
print("Snowball stemmer output for Lehrer:\t",sorted_counts_sstem_lehrer[-10:])

counts_lstem_lehrer = dict(Counter(lehrer_tokenised_lsingles))
sorted_counts_lstem_lehrer = sorted(counts_lstem_lehrer.items(), key=operator.itemgetter(1))
print("Lancaster stemmer output for Lehrer:\t",sorted_counts_lstem_lehrer[-10:])

#top 10 obama
counts_pstem_obama = dict(Counter(obama_tokenised_psingles))
sorted_counts_pstem_obama = sorted(counts_pstem_obama.items(), key=operator.itemgetter(1))
print("Porter stemmer output for Obama:\t",sorted_counts_pstem_obama[-10:])

counts_sstem_obama = dict(Counter(obama_tokenised_ssingles))
sorted_counts_sstem_obama = sorted(counts_sstem_obama.items(), key=operator.itemgetter(1))
print("Snowball stemmer output for Obama:\t",sorted_counts_sstem_obama[-10:])

counts_lstem_obama = dict(Counter(obama_tokenised_lsingles))
sorted_counts_lstem_obama = sorted(counts_lstem_obama.items(), key=operator.itemgetter(1))
print("Lancaster stemmer output for Obama:\t",sorted_counts_lstem_obama[-10:])

#top 10 romney
counts_pstem_romney = dict(Counter(romney_tokenised_psingles))
sorted_counts_pstem_romney = sorted(counts_pstem_romney.items(), key=operator.itemgetter(1))
print("Porter stemmer output for Romney:\t",sorted_counts_pstem_romney[-10:])

counts_sstem_romney = dict(Counter(romney_tokenised_ssingles))
sorted_counts_sstem_romney = sorted(counts_sstem_romney.items(), key=operator.itemgetter(1))
print("Snowball stemmer output for Romney:\t",sorted_counts_sstem_romney[-10:])

counts_lstem_romney = dict(Counter(romney_tokenised_lsingles))
sorted_counts_lstem_romney = sorted(counts_lstem_romney.items(), key=operator.itemgetter(1))
print("Lancaster stemmer output for Romney:\t",sorted_counts_lstem_romney[-10:])

def positiveWords():
	inFile = open("positive.txt", "r")
	positiveFile = inFile.read()
	pstemmer = PorterStemmer()
	words = positiveFile.split()
	posWords = []
	for word in words:	
		posWords.append(pstemmer.stem(word))
	return posWords

poswrds = positiveWords()


#-------LEHER----------
lehrer_topwords = list(Counter(lehrer_tokenised_psingles))
#print(lehrer_topwords)

lehrer_top_positivewords = set(lehrer_topwords).intersection(poswrds)
#print(lehrer_top_positivewords)

#-------OBAMA----------
obama_topwords = list(Counter(obama_tokenised_psingles))
#print(obama_topwords)

obama_top_positivewords = set(obama_topwords).intersection(poswrds)
#print(obama_top_positivewords)

#-------ROMNEY----------
romney_topwords = list(Counter(romney_tokenised_psingles))
#print(romney_topwords)

romney_top_positivewords = set(romney_topwords).intersection(poswrds)
#print(romney_top_positivewords)


lehrerPositiveWords = {}
obamaPositiveWords = {}
romneyPositiveWords = {}

#lehrer top positive words
lehrer_poswordcount = 0
for key,value in counts_pstem_lehrer.items():
	for x in lehrer_top_positivewords:
		if key == x:
			lehrer_poswordcount = lehrer_poswordcount+1
			lehrerPositiveWords[key] = value

sortedLehrerPosWords = sorted(lehrerPositiveWords.items(), key=operator.itemgetter(1))
print("Lehrer positive words: " , sortedLehrerPosWords[-10:], "total:", lehrer_poswordcount)

#obama top positive words
obama_poswordcount = 0
for key,value in counts_pstem_obama.items():
	for x in obama_top_positivewords:
		if key == x:
			obama_poswordcount = obama_poswordcount + 1 
			obamaPositiveWords[key] = value

sortedobamaPosWords = sorted(obamaPositiveWords.items(), key=operator.itemgetter(1))
print("Obama positive words:" , sortedobamaPosWords[-10:] , "total:", obama_poswordcount)

#romney top positive words
romney_poswordcount = 0
for key,value in counts_pstem_romney.items():
	for x in romney_top_positivewords:
		if key == x:
			romney_poswordcount = romney_poswordcount + 1 
			romneyPositiveWords[key] = value

sortedromneyPosWords = sorted(romneyPositiveWords.items(), key=operator.itemgetter(1))
print("Romney positive words:" , sortedromneyPosWords[-10:] , "total:", romney_poswordcount)


with open('topPositiveWords.csv', 'w', newline='') as csvfile:
    Pwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    l = "Lehrer"+str(lehrer_poswordcount)
    o = "Obama"+ str(obama_poswordcount)
    r = "Romney"+ str(romney_poswordcount)
    Pwriter.writerow(" ")
    Pwriter.writerow(l)
    Pwriter.writerow(o)
    Pwriter.writerow(r)


print("Total Positive words of every speaker is logged into topPositiveWords.csv")



