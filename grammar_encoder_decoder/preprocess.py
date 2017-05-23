import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

from bs4 import BeautifulSoup as bs
import pickle
f = open('nucle3.2.p', 'r')
soup = pickle.load(f)

paragraphs = []
for p in soup.textword.findChildren('p'): paragraphs += p
mistakelis = []
for mistake in soup.annotation.findChildren('mistake'):
	mistakelis.append([mistake.correction.text, int(mistake['start_par']) - 1, int(mistake['start_off']), int(mistake['end_off'])])

for p in paragraphs: print p

for mistake in mistakelis:
	para = paragraphs[mistake[1]]
	para = para[:mistake[2]] + mistake[0] + para[mistake[3]:]
	paragraphs[mistake[1]] = para

for p in paragraphs: print p
f.close()
