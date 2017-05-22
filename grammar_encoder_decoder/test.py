from bs4 import BeautifulSoup as bs
import sys
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout)

file = open("nucle3.2.sgml")
soup = bs(file, "lxml")

for para in soup.find_all('p'):
	print para.text
	