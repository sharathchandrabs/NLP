import sys
import time
import string


english_dict  = []
english_words = []
spanish_dict  = []
spanish_words = []
sentence_eng_sp_pairs = [] 
    
probabilities = {}
translation_probs   = {} 
    
count_eng_sp = {} 
total_sp  = {} 
total_s  = {} 

def load_into_dict(dic):
  output_dict  = []
  output_words = []
  f = open(dic, 'r',encoding='Latin-1')
  for line in f.readlines():
    output_dict.append( line.rstrip() )
    output_words = output_words + line.split()
  f.close()
  output_words = list( set(output_words) )
  return output_dict, output_words


def spanish_english_sentence_pairs(english_dict, spanish_dict):
  pair = []
  for i in range(len(english_dict)):
    pr = (english_dict[i], spanish_dict[i])
    pair.append(pr)
  return pair



def initial_probabilities_calc(spanish_words, spanish_dict, english_dict):
  possib = {}
  for word in spanish_words:
    word_possib = []
    for sent in spanish_dict:
      if word in sent:
        inSent = english_dict[ spanish_dict.index(sent) ]
        word_possib = word_possib + inSent.split()
    word_possib = list( set(word_possib))
    possib[word] = word_possib
  return possib



def init_uniform_probabilities_EF(spanish_words, probabilities):
  translation_probs = {}
  for word in spanish_words:
    word_poss = probabilities[word]
    if (len(word_poss)==0):
      print(word, word_poss)
    uniform_prob = 1.0 / len(word_poss)
    word_probabs = dict( [(w, uniform_prob) for w in word_poss] )
    translation_probs[word] = word_probabs
  return translation_probs



def set_word_count_to_zero(spanish_words, probabilities):
  count_eng_sp = {}
  total_sp = {}
  for word in spanish_words:
    word_poss = probabilities[word]
    count_zeroed_down = dict( [(w, 0) for w in word_poss] )
    count_eng_sp[word] = count_zeroed_down
    total_sp[word] = 0
  return count_eng_sp,total_sp






def model_convergence(sentence_eng_sp_pairs,total_s,translation_probs,spanish_words,probabilities):
  converged_status = False
  counter = 0
  while not(converged_status):
    count_eng_sp,total_esp = set_word_count_to_zero(spanish_words, probabilities)
    #   print(count_eng_sp, total_esp)
    for (e_s, f_s) in sentence_eng_sp_pairs:
      e_s_split = e_s.split()
      f_s_split = f_s.split()
      for e in e_s_split:
        total_s[e] = 0
        for f in f_s_split:
          esp_probs = translation_probs[f]
          if (e not in esp_probs):
            continue
          total_s[e] += esp_probs[e]

        for f in f_s_split:
          if (e not in translation_probs[f]):
            continue
          count_eng_sp[f][e] += translation_probs[f][e] / total_s[e]
          total_esp[f] += translation_probs[f][e] / total_s[e]



    for f in spanish_words:
      f_poss = probabilities[f]
      for e in f_poss:
        #print(total_esp[f])
        translation_probs[f][e] = count_eng_sp[f][e] / total_esp[f]
    if (counter>=84):
      converged_status = True
    counter += 1
    print("Iteration", counter, "completed.")
  print("Model has converged.")



def print_sentences(spanish_dict,translation_probs):
  f = open('myTranslation.txt', 'w',encoding='Latin-1')
  print("Writing Translations to myTranslation.txt file. Please wait......")
  for line in spanish_dict:
    translated_english_sentence = ""
    for word in line.split():
      english_word = translation_probs[word]
      items = sorted(iter(english_word.items()), key=lambda k_v: (k_v[1],k_v[0]))
      items.reverse()
      (top, va) = items[0]
      translated_english_sentence+=top+" "

    #print(translated_english_sentence+"\n")
    
    f.write(translated_english_sentence+"\n")
  print("File write complete. Use the myTranslation.txt file to fetch Bleu Score")


def execute_ibm(args,sentence_eng_sp_pairs,probabilities,translation_probs,total_s):
  
  english_dict, english_words = load_into_dict( args[1] )
  spanish_dict, spanish_words = load_into_dict( args[2] )
  
  sentence_eng_sp_pairs = spanish_english_sentence_pairs(english_dict, spanish_dict)

  probabilities = initial_probabilities_calc(spanish_words, spanish_dict, english_dict)
  translation_probs = init_uniform_probabilities_EF(spanish_words, probabilities)
  print("Performing convergence..")
  
  model_convergence(sentence_eng_sp_pairs,total_s,translation_probs,spanish_words,probabilities)
  #output_english_spanish_translation(translation_probs)
  print_sentences(spanish_dict,translation_probs)



def main():
  print("Loading........")
  
  args = sys.argv
  
  execute_ibm(args,sentence_eng_sp_pairs,probabilities,translation_probs,total_s)
  

if __name__=="__main__":
  main()

