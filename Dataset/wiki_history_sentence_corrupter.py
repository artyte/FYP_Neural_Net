import csv
import numpy as np
import nltk
import string
from nltk.stem import WordNetLemmatizer

with open('wiki_history.csv', encoding = "utf8") as f0:
	read = csv.reader(f0)
	row_count = sum(1 for row in read) - 1
	f0.seek(0) #reset readline to start for other reading purposes

	#get a set of uniformly randomly distributed numbers that would be the index of the row to be corrupted
	numlis = np.random.uniform(2,row_count,int(row_count*0.8))
	numdic = {}
	
	#remove duplicate numbers using a dictionary
	for num in numlis:
		if not (int(num) in numdic.keys()):
			numdic[int(num)] = 1	
			
	#put into a numbers from dictionary into a list to allow sorting
	numlis = [k for k in numdic]
	numlis.sort(reverse=True)
	
	#edit numbers in list to reflect number of rows to advance instead of index to be read
	store = numlis[0]
	for index, key in enumerate(numlis):
		numlis[index] = store - key
		store = key
	numlis.reverse()
	numlis.insert(0, store)
	
	with open('wiki_history_write.csv', 'w', encoding = "utf8", newline = '') as f1:
		write = csv.writer(f1)
		
		#sentence to be corrupted is assumed to have verbs to be corrupted
		tostem_tagset = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
		lemma = WordNetLemmatizer()
		for key in numlis:
			for row in read:
				if (key == 1): #if no more rows to advance
					text = nltk.word_tokenize(row[2])
					
					#since pos_tag produces a list of tuples, convert it to list of list
					taglisOftuple = nltk.pos_tag(text) 
					taglisOflis = [list(elem) for elem in taglisOftuple]
					
					#words (corrupted or not) to be placed back in the order that they were in
					taglis = [] 
					for word in taglisOflis:
						if word[1] in tostem_tagset: #if word is to be corrupted
							taglis.append(lemma.lemmatize(word[0], pos = 'v'))
						else: #if word is not be corrupted
							taglis.append(word[0])
							
					#from list to a string
					row[2] = "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in taglis]).strip()
					
					row.append(0) #corrupted sentence given a 0 tag
					write.writerow(row)
					break
				else:
					key -= 1
					row.append(1) #corrupted sentence given a 1 tag
					write.writerow(row)