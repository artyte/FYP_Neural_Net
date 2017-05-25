from bs4 import BeautifulSoup as bs
import sys
import codecs
import pickle
reload(sys)
sys.setdefaultencoding('utf8')

def calculateShift(shift_influ, word_len, start_index, end_index):
	shift = word_len - (end_index - start_index)
	accu = 0
	i = 0
	for obj in shift_influ:
		if start_index < obj[0]:
			shift_influ.insert(i, [start_index, shift])
			break
		else:
			i += 1
			accu += obj[1]

	if i == len(shift_influ): shift_influ.insert(i, [start_index, shift])
	return accu + 1

def efficientNucle():
	sys.setrecursionlimit(10000000)
	soup = bs(open("nucle3.2.sgml"), "lxml")
	f = open('nucle3.2.p', 'w')
	pickle.dump(soup, f)
	f.close()

def editParagraphs(soup):
	import nltk
	import re
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	
	paragraphs = []
	original = []
	for p in soup.textword.findChildren('p'): paragraphs += p	
	for p in paragraphs: 
		p = re.sub('\n', '', p)
		original.append(str(p))
	
	mistakelis = []
	for mistake in soup.annotation.findChildren('mistake'):
		mistakelis.append([mistake.correction.text, int(mistake['start_par']) - 1,
						int(mistake['start_off']), int(mistake['end_off'])])

	shift_influ = []
	prev_par = 0
	for m in mistakelis:
		if m[1] != prev_par: shift_influ = []
		prev_par = m[1]
		p = paragraphs[m[1]]
		shift = calculateShift(shift_influ, len(m[0]), m[2], m[3])
		p = p[:shift + m[2]] + m[0] + p[shift + m[3]:]
		paragraphs[m[1]] = p

	sentences = []
	for p in paragraphs:
		p = re.sub(' +', ' ', p)
		p = re.sub(' \.', '.', p)
		p = re.sub('\n', '', p)
		sentences.append(str(p))
	
	zipped = zip(original, sentences)
	return zipped

def makeCSV():
	import csv
	f = open('nucle3.2.p', 'r')
	soup = pickle.load(f)
	f.close()

	senlis = []
	for doc in soup.findChildren('doc'): senlis += editParagraphs(doc)
	
	f = open('nucle3.2.csv', 'w')
	write = csv.writer(f)
	for row in senlis: write.writerow(row)

makeCSV()
