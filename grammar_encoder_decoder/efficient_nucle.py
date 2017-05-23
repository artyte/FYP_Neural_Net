from bs4 import BeautifulSoup as bs
import sys
import pickle
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.setrecursionlimit(10000000)
soup = bs(open("nucle3.2.sgml"), "lxml")
f = open('nucle3.2.p', 'w')
pickle.dump(soup, f)
f.close()
