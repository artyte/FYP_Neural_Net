import csv
import numpy as np
import nltk
import string
from nltk.stem import WordNetLemmatizer

with open('wiki_history.csv', encoding = "utf8") as f0:
	row_count = len(f0.readlines())
	f0.seek(0)
	read = csv.reader(f0)

	numlis = np.random.uniform(2,row_count,int(row_count*0.8))
	numdic = {}
	for num in numlis:
		if not (int(num) in numdic.keys()):
			numdic[int(num)] = 1	
	numlis = [k for k in numdic]
	numlis.sort(reverse=True)
	store = numlis[0]
	for index, key in enumerate(numlis):
		numlis[index] = store - key
		store = key
	numlis.reverse()
	numlis.insert(0, store)
	print(len(numlis))
	
	with open('wiki_history_write.csv', 'w', encoding = "utf8", newline = '') as f1:
		write = csv.writer(f1)
		
		tostem_tagset = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
		lemma = WordNetLemmatizer()
		for key in numlis:
			for row in read:
				if (key == 1):
					text = nltk.word_tokenize(row[2])
					taglisOftuple = nltk.pos_tag(text)
					taglisOflis = [list(elem) for elem in taglisOftuple]
					taglis = []
					for word in taglisOflis:
						if word[1] in tostem_tagset:
							taglis.append(lemma.lemmatize(word[0], pos = 'v'))
						else:
							taglis.append(word[0])
					row[2] = "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in taglis]).strip()
					row.append(0)
					write.writerow(row)
					break
				else:
					key -= 1
					row.append(1)
					write.writerow(row)