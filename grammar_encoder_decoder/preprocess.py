import sys
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout)

from bs4 import BeautifulSoup as bs
soup = bs(open("nucle3.2.sgml"), "lxml")

print soup.doc
