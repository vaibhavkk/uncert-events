# this file converts file into words 
fd = open('./data/cricket/043.txt')
lines =  fd.readlines()

words = []
for w in lines : 
    wl =  w.split(' ')
    words = words + wl
sample_space = len(set(words))
print("total words in file are , ", len(words) ," with unique words  " , sample_space ) 

fd.close()

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

stopwords = set(stopwords.words('english'))
words = []
import nltk
from collections import Counter
i = 0
import numpy as np
#
metadata_sentencewise = []
all_filtered_sentences = []
# nouns and adjectives tag list 
noun_tags = ['NN','NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'POS', 'WP', 'WP$', 'JJ', 'JJR','JJS']
# verbs and adverbs list 
verb_tags = ['RB', 'RBR','RBS', 'RP','VB','VBD', 'VBG','VBP','VBZ','WRB']
for l in lines : 
	i = i+1 
	word_tokens = word_tokenize(l)
	#filtered_sentence = [w.lower() for w in word_tokens if w not in stopwords and  w.isalnum()]
	tagged_sentence = nltk.pos_tag(word_tokens)
	word_tokens =  [w[0] for w in tagged_sentence if w[1] in verb_tags]
	filtered_sentence = [w.lower() for w in word_tokens if  w.isalnum()== True]
	all_filtered_sentences.append(filtered_sentence)
	words  = words + filtered_sentence 
al = []
for e in all_filtered_sentences : 
    s = " "
    for a in e : 
        s = s+ " " +a
    al.append(s)


def get_ngrams_stat(n,w,arg_lines) :
# arguments 
# n = number of grams 
# w = words which will form n-grams 
# l = list of lines in which n-grams are searched 

	import nltk
	from nltk.util import ngrams 
	grams_list = []
	u = list(ngrams(w,n))
	for e in u :
		s = ""
		for i in e :
			s = s+ " " +i
		for l  in arg_lines :
			if s in l.lower() :
				#print(s, l.lower())
				grams_list.append(s)
	from collections import Counter
	c = Counter(grams_list)
	for k,v in c.items() :
		print(k, ",",v)
	print("n is ", n," Total unique grams",  len(c), " for grams  ", len(u) , " in words ", len(l) ) 

get_ngrams_stat(1,words,al)
get_ngrams_stat(2,words,al)
get_ngrams_stat(3,words,al)
get_ngrams_stat(4,words,al)
get_ngrams_stat(5,words,al)

'''
import nltk
		

b = list( nltk.bigrams(words))


#################
#cacluate entropy for bigrams 


from nltk.util import ngrams 
unigrams_list = [] 
u = list(ngrams(words,1))
for e in u : 
     s = e[0]
     for l  in lines : 
         if s in l.lower() : 
             #print(s, l.lower())
             unigrams_list.append(s)
from collections import Counter
c = Counter(unigrams_list)
for k,v in c.items() : 
	print(k, ",",v)

print("Total unique unigrams",  len(c), " for unigrams  ", len(u) , " in words ", len(words) )


bigram_list = []
for e in b : 
    s = e[0] + " " + e[1]
    for l  in lines : 
        if s in l.lower() : 
            #print(s, l.lower())
            bigram_list.append(s)
from collections import Counter
c = Counter(bigram_list)
#print(c.items())
for k,v in c.items() :
	print(k, ",",v)

print("Total unique bigrams",  len(c), " for bigrams ", len(b), " in words" ,len(words))

t = list(ngrams(words,3))
trigrams_list = []
for e in t : 
	s = e[0] + " "+e[1]+" "+e[2] 
	for l in lines : 
		if s in l.lower() : 
			trigrams_list.append(s)
from collections import Counter
c = Counter(trigrams_list)
#print(c.items())
for k,v in c.items() :
	print(k, ",",v)

print("Total unique trigrams",  len(c), " for  trigrams ", len(t), "in words", len(words))
q = list(ngrams(words,4))
q_list = [] 
for e in q : 
	s = e[0] + " "+e[1]+" "+e[2]+ " " +e[3]
	for l in lines :
		if s in l.lower() :
			q_list.append(s)
from collections import Counter
c = Counter(q_list)
#print(c.items())
for k,v in c.items() :
	print(k, ",",v)


print("Total unique quadgrams",  len(c), " for quadgrams  ", len(q) , " in words" , len(words))


fn = list(ngrams(words,5))

#fn_list = []
#for e in fn :  
#	s = e[0] + " "+e[1]+" "+e[2]+ " " +e[3] + e[4]
#	for l in lines : 
#		if s in l.lower() : 
#			fn_list.append(s)
#from collections import Counter 
#c = Counter(fn)
#for k,v in c.items : 
#	print(k, ",", v)

#print("Total Five grams",  len(c), " for total words ", len(words))
'''


