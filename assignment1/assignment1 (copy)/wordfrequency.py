def wordfrequency(text):
	wordlist = []
	freqDict = {}
	words = text.split()
	for word in words:
		wordlist.append(words.count(word))
		freqDict[word] = words.count(word)
	print(freqDict)
sentence = "she sells sea shells by the sea shore"
wordfrequency(sentence)
