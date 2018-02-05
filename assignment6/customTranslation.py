
import re
from nltk import UnigramTagger, BigramTagger, TrigramTagger, HiddenMarkovModelTagger
from nltk.corpus import cess_esp
import pickle
import os.path
from nltk.corpus import brown
from nltk import UnigramTagger





def readCorpus(filename):
	input_corpus = []
	with open(filename) as f:
		for line in f:
			sentences = line.split(":")
			if (sentences):
				input_corpus.append(sentences)
	return input_corpus



devCorpus = readCorpus("dev.txt")
testCorpus = readCorpus("test.txt")

testCorpus = devCorpus + testCorpus


#---------------------------------------------------------------------------Read Corpus---------------------------------------------------------------------------------------

        

        
    #Generate Spanish-Tags file
    # def printSpanishTags(self):
    #     sents = cess_esp.tagged_sents()
    #     tagger = HiddenMarkovModelTagger.train(sents)
        
    #     fullCorpus = self.fullCorpus()
    #     tagsDictionary = dict()
    #     for line in fullCorpus:
    #         spanishSentence = line[0]
    #         spanishTokens = re.compile('\W+', re.UNICODE).split(unicode(spanishSentence, 'utf-8'))
    #         tags = tagger.tag(spanishTokens)
    #         for idx, token in enumerate(spanishTokens):
    #             if (len(token) > 0):
    #                 tag = tags[idx][1]
    #                 sys.stdout.write(token.encode('utf-8'))
    #                 sys.stdout.write(":")
    #                 sys.stdout.write(tag)
    #                 sys.stdout.write("\n")
        
def generate_spanish_tags():
	postags = None
	if not postags:
		postags = dict()
		file = open("Espanol-Tags.txt", "r")
		for l in file:
			token = l.split(":")
			wrd = str(token[0])
			tag = str(token[1])
			postags[wrd] = tag
		postags = postags
	return postags







#--------------------------------------------------------------------------Read Corpus-----------------------------------------------------------------------------------------


#---------------------------------------------------------------------Initialize dictionary----------------------------------------------------------------------------------------

dictionary = dict()
english_words_set = set()

def readDictionary():
	with open("esp-eng-dictionary.txt") as f:
		for l in f:
			words = l.split(":")
			spanish_word = str(words[0])
			english_word = None
			if (len(words) > 1):
				english_word = words[1].strip("\n")
				tmpList = english_word.split(",")
			dictionary[spanish_word] = tmpList
			for word in tmpList:
				english_words_set.add(word)
	return dictionary, english_words_set

dictionary,englishWordSet = readDictionary()


def train_unigram_tagging():
	tagged_sentences = brown.tagged_sents()
	unigram_tagging = UnigramTagger(tagged_sentences)
	return unigram_tagging

brown_unigram_tagging = train_unigram_tagging()


def english_POS_dict():
	englishDict = dict()
	newList = []
	for word in englishWordSet:
		newList.append(word)
	word_pos = brown_unigram_tagging.tag(newList)
	for tup in word_pos:
		englishDict[tup[0]] = tup[1]
	return englishDict




englishPosDict = english_POS_dict()


def spanish_word_to_english_words(spanishWord):
	list_of_words = dictionary.get(spanishWord.lower(), None)
	return list_of_words
    

def english_word_for_spanish_word(token):
	#Nouns
	#print(token)
	noun = ('NN', 'NNS', 'NNP', 'NNPS', 'NR');
	noun_plural = ('NNS', 'NNPS');
	#Verb Mode
	verb_imperative = ('BE', 'DO', 'HV', 'VB');
	verb_infinitive = ('BE', 'DO', 'HV', 'VB');
	verb_gerund = ('BEG', 'HVG', 'VBG');
	verb_participle = ('BEG', 'BEN', 'HVG', 'HVN', 'VBG', 'VBN');
	#Verb Time
	verb_present = ('VB','VBG', 'VBP', 'VBZ', 'BEG', 'BEM', 'BER', 'BEZ', 'DO', 'DOZ', 'HV', 'HVG', 'HVZ');
	verb_past = ('VBD', 'VBN', 'BED', 'BEDZ', 'BEN', 'DOD', 'HVD', 'HVN');
	#Verb Person
	verb_first_person = ('BED', 'BEDZ', 'BEM', 'BER');
	verb_second_person = ('BED', 'BER');
	verb_third_person  = ('BED', 'BEDZ', 'BEZ', 'DOZ', 'HVZ', 'VBZ');
	#Verb Number
	verb_singular  = ('BED', 'BEDZ','BER', 'BEM', 'BEZ', 'DOZ', 'HVZ', 'VBZ');
	verb_plural = ('BED', 'BER');
	#Adjective
	adjective = ('JJ', 'JJ$', 'JJR', 'JJS', 'JJT');
	conjunction = ('CC', 'CS', 'ABX', 'DTX');
	pronoun = ('ABL', 'ABN', 'ABX', 'AP', 'AP$', 'DT', 'DT$', 'DTI', 'DTS', 'DTX', 'PN', 'PN$', 'PP$', 'PP$$', 'PPL', 'PPLS', 'PPO', 'PPS', 'PPSS', 'WP$', 'WPO', 'WPS');
	pronounNominative = ('PPS', 'PPSS', 'WPS');
	
	spanish_word = token['originalToken']
	spanish_tag = token['spanish_POS']

	word_list = dictionary.get(spanish_word.lower(), None)
	chosenEnglishWord = None
	if word_list and spanish_tag:
		spanishWordCategory = spanish_tag[0]
		final_word_list = word_list[:]
		for english_word in word_list:
			
			if len(final_word_list) == 1:
				break
			english_tag = englishPosDict.get(english_word, None)
			if spanishWordCategory == 'n':
				
				if english_tag not in noun and english_word in final_word_list:
					final_word_list.remove(english_word)
				else:
					
					pluralTag = spanish_tag[3]
					if (pluralTag == 'p'):
						
						if english_tag not in noun_plural and english_word in final_word_list:
							final_word_list.remove(english_word)
			elif spanishWordCategory == 'v':
				
				if spanish_tag[2] != 'g':
					if english_tag in verb_gerund and english_word in final_word_list:
						final_word_list.remove(english_word)
				
				if spanish_tag[2] == 'm':
					if english_tag not in verb_imperative and english_word in final_word_list:
						final_word_list.remove(english_word)
				
				if spanish_tag[2] == 'n':
					
					if english_tag not in verb_infinitive and english_word in final_word_list:
						final_word_list.remove(english_word)

				tenseTag = spanish_tag[3]

				if tenseTag == 'p' or tenseTag == '0':

					if english_tag not in verb_present and english_word in final_word_list:

						final_word_list.remove(english_word)
				if tenseTag == 'f':

					if english_tag not in verb_infinitive and english_word in final_word_list:
						final_word_list.remove(english_word)
				if tenseTag == 's':

					if english_tag not in verb_past and english_word in final_word_list:
						final_word_list.remove(english_word)
				personTag = spanish_tag[4]
				if personTag == '1':
					if english_tag not in verb_first_person and english_word in final_word_list:
						if english_word in final_word_list: final_word_list.remove(english_word)
				if personTag == '2':
					if english_tag not in verb_second_person and english_word in final_word_list:
						if english_word in final_word_list: final_word_list.remove(english_word)
				if personTag == '3':
					if english_tag not in verb_third_person and english_word in final_word_list:
						if english_word in final_word_list: final_word_list.remove(english_word)

				numberTag = spanish_tag[5]
				if numberTag == 's':
					if english_tag not in verb_singular and english_word in final_word_list:
						if english_word in final_word_list: final_word_list.remove(english_word)
				else:
					if english_tag not in verb_plural and english_word in final_word_list:
						if english_word in final_word_list: final_word_list.remove(english_word)
			elif spanishWordCategory == 'a':

				if english_tag not in adjective and english_word in final_word_list:
					final_word_list.remove(english_word)
			elif spanishWordCategory == 'c':

				if english_tag not in conjunction and english_word in final_word_list:
					final_word_list.remove(english_word)
			elif spanishWordCategory == 's':

				if english_tag != 'IN' and english_word in final_word_list:
					final_word_list.remove(english_word)
			elif spanishWordCategory == 'p' and spanish_tag[5] == 'n':

				if english_tag not in pronounNominative and english_word in final_word_list:
					final_word_list.remove(english_word)

		if chosenEnglishWord == None:
			if len(final_word_list) > 0:
				chosenEnglishWord = final_word_list[0].strip()
			else:
				chosenEnglishWord = word_list[0]
	elif (spanish_tag == None) and word_list:

		chosenEnglishWord = word_list[0]
	chosenEnglishWordTag = englishPosDict.get(chosenEnglishWord, None)
	token['english_POS'] = chosenEnglishWordTag



	return chosenEnglishWord
    
















#---------------------------------------------------------------------Initialize dictionary----------------------------------------------------------------------------------------

def sentence_translation(foreignSentence):
	translated_tokens = []
	translated_sent = ""
	spanish_tokens = re.compile('(\W+)', re.UNICODE).split(str(foreignSentence))

	for token in spanish_tokens:
		#print (token)
		translated_words = spanish_word_to_english_words(token)
		if translated_words:
			translated_word = translated_words[0]
			translated_tokens.append(translated_word)
		else:
			translated_tokens.append(token)

	for token in translated_tokens:
		translated_sent = translated_sent + token
	return translated_sent







for sent_pairs in testCorpus:
	spanish_sent = sent_pairs[0]
	actual_english_sent = sent_pairs[1]
	dict_list_token = []
	modified_sentence = ""


	#direct translation section
	direct_translation = sentence_translation(spanish_sent)




	#modified translation section
	spanish_words = re.compile('(\W+)', re.UNICODE).split(str(spanish_sent))
	spanish_words.pop()
	#print(spanish_words)
	for ids, token in enumerate(spanish_words):
		#print(ids,token)
		token_dictionary = dict()
		token_dictionary['originalToken'] = token
		token_dictionary['spanish_POS'] = generate_spanish_tags().get(token, None)

		if (len(token) > 0):
			if token[0].isupper():
				token_dictionary['upper'] = True
			else:
				token_dictionary['upper'] = False
		else:
			token_dictionary['upper'] = False
		dict_list_token.append(token_dictionary)

	for spanishToken in dict_list_token:
		original = spanishToken['originalToken']
		translated = english_word_for_spanish_word(spanishToken)
		if translated:
			spanishToken['translatedToken'] = translated
		else:
			spanishToken['translatedToken'] = original
	for token in dict_list_token:
            translatedToken = token['translatedToken']

            modified_sentence = modified_sentence + translatedToken

	
	print ("Spanish: " + spanish_sent)
	print ("Direct Translation: " + direct_translation)
	print ("Custom Model Translation: " + modified_sentence)
	print ("Human Translation: " + actual_english_sent)