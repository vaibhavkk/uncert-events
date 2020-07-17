#this is python code to process all files to calculate uncertainty 
import sys
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import load_files

''' create a cluster based on entropy ''' 
def get_cluster_stats(list_data_1d) : 
	''' list_data_1d is the data which is to be clustered, it is typically 1D array of numbers 
	'''
	cluster_data = []
	for d in list_data_1d : 
    		for e in d : 
        		cluster_data.append(e[1])
	from sklearn.cluster import KMeans
	import numpy as np
	X = np.array(cluster_data)
	X = X.reshape(-1,1)
	kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
	l = kmeans.labels_
	#for  i in  range(0,len(cluster_data)) : 
	#    print(cluster_data[i] , "# ", l[i])
	from sklearn import metrics
	print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, l, sample_size=100))



def get_ngrams_stat(n,w,arg_lines) :
# arguments 
# n = number of grams 
# w = words which will form n-grams 
# l = list of lines in which n-grams are searched 
	#print(n, " in get stats\n", w , "word list \n",arg_lines ," arg lines")
# returns entropy of all words 		
	import nltk
	from nltk.util import ngrams 
	grams_list = []
	u = list(ngrams(w,n))
#	print(u, '\n%%%%%%%%%%%\n')
	for e in u :
		s = ""
		for i in e :
			s = s+ " " +i
		#for l  in arg_lines : if arg_lines is list uncomment this 
		#	print(l , " in arg lines")
			if s in arg_lines :
				#print(s, l.lower())
				#print(s, "add")
				grams_list.append(s)
	#print(grams_list, "\n *******\n" )
	from collections import Counter
	c = Counter(grams_list)
	import numpy as np 
	#print("n is ", n," Total unique grams",  len(c), " for grams  ", len(u) , " total words ", len(arg_lines) ) 
	unique_unigrams  = len(c)
	w_ig = []
	dem = []# this is document entropy matrix 	
	for k,v in c.items() :
		e = -(v/unique_unigrams)*np.log(v/unique_unigrams)
#		print(k, ",",v , ", ", e)
		w_ig.append(e*v)
		dem.append((k,e))
	print( "n is ", n," Total unique grams",  len(c), " for grams  ", len(u) , " total words ", len(arg_lines), np.sum(w_ig)/unique_unigrams, " information gain" , 1-np.sum(w_ig)/unique_unigrams, " uncertainty" )	
	return dem 

print("Reading BBC Sports data ")

newgroupdata = load_files("../cvm_code/bbcsport/",
            description = None, load_content = True,
            encoding='latin1', decode_error='strict', shuffle=False, random_state=42)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
en_stop = stopwords
tokenizer = RegexpTokenizer(r'\w+')
import numpy as np
i = 0
# nouns and adjectives tag list 
noun_tags = ['NN','NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'POS', 'WP', 'WP$', 'JJ', 'JJR','JJS']
# verbs and adverbs list 
verb_tags = ['RB', 'RBR','RBS', 'RP','VB','VBD', 'VBG','VBP','VBZ','WRB']
dem = [] # document entropy matrix 
for f in newgroupdata.data :
	print("_________________",i,"________________") 
	i= i+1
	text = f 
	text.lower()
	text1 = word_tokenize(text)
	filtered_sentence = [w.lower() for w in text1 if w not in stopwords and  w.isalnum()]
	non_filtered_sentence = [w.lower() for w in text1 ]

	tagged_sentence = nltk.pos_tag(text1)
	verb_word_tokens =  [w[0] for w in tagged_sentence if w[1] in verb_tags]
	noun_word_tokens =  [w[0] for w in tagged_sentence if w[1] in noun_tags]
	all_words = []
	sentence = " "
	
#	for e in text.split() : # if text is list 
	for e in filtered_sentence  : # for stop words 
		if e not in en_stop : 
			all_words.append(e)
			sentence = sentence+ " " +e
	all_words_f = []
	sentence_f = " "	
	for f in non_filtered_sentence :	
		all_words_f.append(f)
		sentence_f = sentence_f+ " " +f	
	all_words_v = []
	sentence_v = " "	
	for v in verb_word_tokens :	
		all_words_v.append(v)
		sentence_v = sentence_v+ " " +v	
	all_words_n  = []
	sentence_n = " "	
	for n in noun_word_tokens :	
		all_words_n.append(n)
		sentence_n = sentence_n+ " " +n	
	
	#print(sentence)
	# create unigram of the data in file i.e of sentence
	# 
	'''
	import nltk
	from nltk.util import ngrams 
	grams_list = []
	u = list(ngrams(all_words,2))
	for e in u : 
		if e[0] in sentence : 
			print (e)
	'''
	
#	dem.append(get_ngrams_stat(1,all_words,sentence))	
#	dem.append(get_ngrams_stat(2,all_words,sentence))
#	dem.append(get_ngrams_stat(3,all_words,sentence))
#	dem.append(get_ngrams_stat(4,all_words,sentence))
#	dem.append(get_ngrams_stat(5,all_words,sentence))
#	print("Without stop word removal ")
#	dem.append(get_ngrams_stat(1,all_words_f,sentence_f))
#	dem.append(get_ngrams_stat(2,all_words_f,sentence_f))
#	dem.append(get_ngrams_stat(3,all_words_f,sentence_f))
#	dem.append(get_ngrams_stat(4,all_words_f,sentence_f))
#	dem.append(get_ngrams_stat(5,all_words_f,sentence_f))
#	print("With verbs and stop words ")
#	dem.append(get_ngrams_stat(1,all_words_v,sentence_v))
#	dem.append(get_ngrams_stat(2,all_words_v,sentence_v))
#	dem.append(get_ngrams_stat(3,all_words_v,sentence_v))
#	dem.append(get_ngrams_stat(4,all_words_v,sentence_v))
#	dem.append(get_ngrams_stat(5,all_words_v,sentence_v))
#	print("With nouns and stop words ")
#	dem.append(get_ngrams_stat(1,all_words_n,sentence_n))
#	dem.append(get_ngrams_stat(2,all_words_n,sentence_n))
#	dem.append(get_ngrams_stat(3,all_words_n,sentence_n))
#	dem.append(get_ngrams_stat(4,all_words_n,sentence_n))
	dem.append(get_ngrams_stat(5,all_words_n,sentence_n))
	
#print (dem)
get_cluster_stats(dem)


