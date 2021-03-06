# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from re import search,findall
from re import compile as recomp
#from yara import compile as comp
#import ahocorasick
from sys import argv, stdout, stdin
#from requests import get as uget
from numpy import exp,loadtxt, asarray, dot
#from numpy import append as npappend
from lxml.html import parse as lparse
from urllib2 import urlopen
from urlparse import urlparse
from time import time
#from concurrent.futures import ThreadPoolExecutor
from whois import whois
from datetime import datetime
from threading import Thread
import pythonwhois,time
'''
# used for debugging code
import sys,logging,optparse
LOGGING_LEVELS = {'critical': logging.CRITICAL,
				  'error': logging.ERROR,
				  'warning': logging.WARNING,
				  'info': logging.INFO,
				  'debug': logging.DEBUG}
parser = optparse.OptionParser()
parser.add_option('-l', '--logging-level', help='Logging level')
parser.add_option('-f', '--logging-file', help='Logging file name')
(options, args) = parser.parse_args()
logging_level = LOGGING_LEVELS.get(options.logging_level, logging.NOTSET)
logging.basicConfig(level=logging_level, filename=options.logging_file,
format='%(asctime)s %(levelname)s: %(message)s',
datefmt='%Y-%m-%d %H:%M:%S')
'''
# Global regex need to loaded to memory
URL_REGEX = r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))"""#url not too sensitive
URL_REGEX2 = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

URL_REGEX3 = r"""@^(https?|ftp)://[^\s/$.?#].[^\s]*$@iS"""

#This regex is for ip url pattern
ippattern = '(?:http.*://)?(?P<ip>([0-9]+)(?:\.[0-9]+){3}).*'
#This regex is for domain and port pattern
urlpattern = '(?:http.*://)?(?P<domain>[^:/ ]+).?(?P<port>[0-9]*).*'

#Global white list
whitelistdomain= ['google.com','facebook.com','twitter.com','youtube.com','wordpress.org','linkedin.com','sedo.com','sedoparking.com',
'wikipedia.org','blogspot.com','yahoo.com','ebay.com','paypal.com']

#Global Yara rules At very first: load yara rules to the memory, we only do this for once; and we will use it all the time
#Global weights and bias for NN cliassifier
def getweights_bias(layers=5):
	weights = []
	for x in range(layers):
		weights.append(loadtxt('./weights.out'+str(x),delimiter=','))
	bias = loadtxt('./bias.out',delimiter=',')
	return weights,bias
weights,bias = getweights_bias()

#function for find short version url without http:// or https://
def shorturl(wholeurl):
	pos = wholeurl.find("//")+1
	return wholeurl[pos:]

#function for generate feature 1, whether the address is ip addressed.
def ipisnum(wholeurl): #feature 1
	ipmatch = search(ippattern, wholeurl)
	return 1 if ipmatch else -1

#Feature 3 check whether URL is redirected.
def urlredirect(wholeurl,page):
	try:
		#if not wholeurl.startswith("http://") or wholeurl.startswith("https://"):
		#	wholeurl = "http://" + wholeurl
		return -1 if (page.geturl()) == wholeurl else 1
	except Exception as e:
		return -1# this should not be the case if the whole url is invaid	
#feature 9 use whois call to get dns

def f9whois(w,today):
		try:
			if w:
				return -1 if w['expiration_date'][0] > today+timedelta(days=365) else 1
			else:		
				return 1
		except Exception as e:
			#print("f9whois: " + str(e))
			return -1
#feature 24 DNS record less than 6 months
def f24whois(w,today):
		try:
			if w:
				return -1 if w['creation_date'][0] < today-timedelta(days=183) else 1
			else:		
				return 1
		except Exception as e:
			#print("f9whois: " + str(e))
			return -1


#feature 10 find the url of favicons and compare the domain with url domain	
def favicondomain(wholeurl,domain,page):
	try:
		doc = lparse(page)
	except Exception as e:
		print("FAVICONDOMAIN ERROR: " + str(e))
		return -1
	favicons = doc.xpath('//link[@rel="shortcut icon"]/@href')

	if len(favicons) > 0:
		if(favicons[0].startswith("/")):
			favicons[0] = domain + '' + favicons[0]
		favicon = favicons[0]
	else:
		return 1
	fdomain = shorturl(favicon).split('/')[0]
	if fdomain == domain:
		return -1
	else:
		return 1

#feature 11: check the port numer
def domain_port_num(wholeurl):#feature port number
	m = search(urlpattern,wholeurl)
	port = m.group('port')
	domain = m.group('domain')
	portnum = 80 if not port else int(port)
	return domain,portnum

#feature 13: check the request domain
def RequestURL(html,domain):
	try:
		urllist = findall(URL_REGEX,html)
	except Exception as e:
		return -1
	if not urllist:
		return -1
	total = len(urllist)
	legcut = 0.22 * total
	phishingcut = 0.61 * total
	count = 0
	for url in urllist:
		m = search(urlpattern,url)
		urldomain = m.group('domain')
		if domain != urldomain:
			count += 1
		if count > phishingcut:
			return 1
	if count < legcut:
		return -1
	else:
		return 0

#feature 14: anchor URL not point to same page

def anchorURL(html):
	anchorlist = findall('<a href="(?P<anchor>.*)">',html)
	count = 0	
	total = len(anchorlist)
	for anchor in anchorlist:
		if '#' in anchor:
			count += 1
		elif '::void(0)' in anchor:
			count += 1
	if count > 0.67*total:
		return 1
	elif count < 0.31*total:
		return -1
	else:
		return 0


#feature 15:  url or path in meta link and script

#a check pathlist function for checking whether this is a path of own domain or an url to other domains.
def checkpathdomain(pathlist,domain):
	count = 0
	try:
		for path in pathlist:
			if path.startswith('http://') or path.startswith('https://') or path.startswith('//'):
				count += 1
			elif path.startswith('.') or path.startswith('/'):
				continue
			else:
				urldomain = path.split('/')[0]
				if urldomain == domain:
					continue
				else:
					count += 1
	except Exception as e:
		print("check path",e)
		pass
	return count


#feature 15
def MSLtags(html,domain):
	
	metaurllist = findall('<meta.*(?!>)URL=(.*)".*>', html)
	scripturllist = findall('<script.*(?!>)src="(.*)">', html)
	linklist = findall('<link.*(?!>)href="(.*)">',html)
	total = len(metaurllist) + len(scripturllist) + len(linklist)
	count = checkpathdomain(metaurllist,domain) + checkpathdomain(scripturllist,domain) +checkpathdomain(linklist,domain)
	return -1 if count < 0.17*total else 1 if count > 0.81 * total else 0

#feature 16 1.2.4. Server Form Handler (SFH)

def SFH(formhandle):
	
	for form in formhandle:#phishing form handler is very easy
		if form.startswith("about:blank"):
			return 1
		elif form.startswith('http://') or form.startswith('https://') or form.startswith('//'):
			return 0
	return -1
#feature 17
		
#using mail() or mailto

def Formmail(formhandle):
	for form in formhandle:
		if 'mail(' in form or 'mailto:' in form: #mail or mailto in form action
			return 1
	return -1

#feature 18
#URL in the identity

def identity(w,name):
	try:
		if w['raw'][0]:
			DomainName = findall('.*Domain Name:(.*)', w['raw'][0])
			if	DomainName[0]:
				return -1 if name in DomainName[0] else 1
		return 1
	except Exception as e:
		print("identity", e)
		return 1
#feature 20
# on mouse over
def onmouseover(html):
	mouseaction = findall('onmouseover="([^ >"]*)"', html)
	for ac in mouseaction:
		if 'status' in ac:
			return 1

	return -1
#feature 21
def disableightclick(html):
	return 1 if "event.button==2" in html else -1

#active function sigmoid function for NN classifier. This function will return a value between 0 to 1, which is widely used in probability cacualtion.
def sigmoidf(x,deri=False):
	if(deri == True):
		return x*1.0*(1.0-x)
	return 1.0/(1+exp(-x))
 
#NN classifier, this fucntion take the features vector and compute the output by go throught each layer of weights and bias. it is computed by numpy.dot matrix muliply fucntion. Check numpy.dot if you are interested. The final out put is a classfication value "1" or "-1"
def NNclassifier(features,layers=5):#the prediction function
	ldata = []
	ldata.append(features)
        try:
	     for x in range(layers):
	         if(x==1):
	               dropweight = asarray(weights[x])*0.5
                       ldata.append(sigmoidf(dot(ldata[x],dropweight)+bias[x]))
                 else:
                       ldata.append(sigmoidf(dot(ldata[x],weights[x])+bias[x]))
        except Exception as e:
            print(e)
        return 1 if ldata[-1]>=0.5 else -1 # 1 for phishing, -1 not phishing

# This function take a url as input, output a features vector. This is the most important part of this program.
# The features is computed by the functions above marked features and yara rules.
def urlfeatureextractor(wholeurl):
	#Trules = comp('./TURL.yar')
	#Frules = comp('./FURL.yar')
	#Drules = comp('./domain.yar')
        
	wholeurl = wholeurl.lower()
	if wholeurl.startswith("http://") or wholeurl.startswith("https://"):# This is how to deal with url with out http title
		surl = shorturl(wholeurl)
		try:
			page = urlopen(wholeurl,timeout=1)
			html = page.read()
		except Exception as e:
			page = None
			html = ''
		#hmark = True
	else:
		surl = wholeurl
		wholeurl = "http://"+wholeurl
		try:
			page = urlopen(wholeurl,timeout=1)
			html = page.read()
		except Exception as e:
			page = None
			html = ''
		#hmark = False
	length=len(surl)
	domain,portnum = domain_port_num(wholeurl)
	#if hmark: #hmark is used for which url the input is. if begin with http(s):// mathc Trule, else Frule
	#	rules = Trules
	#else:
	#	rules = Frules
	#dr = Drules

	features = asarray([0]*30)#allocation 30 features
	try:
		#matches = rules.match(data=wholeurl)#match rules for url
		#domainmatches = Drules.match(data = domain) #match rules for domain

		#URL section start
		features[0] = ipisnum(wholeurl) # feature 1 check whether domain is ip
		features[1] = 1 if length>75 else -1 if length<54 else 0 # feature 2 check length
		length =None
		
		features[3] = 1 if "@" in surl else -1#feature 4
		features[4] = 1 if "//" in surl else -1# feature 5
		features[5] = 1 if "-" in domain else -1#feature 6
		dotcount = domain.count('.')
		features[6] = 1 if  dotcount > 3 else 0 if dotcount == 3 else -1# feature 7
		dotcount = None
		features[7] = -1 if wholeurl[:8] == "https://" else 1#feature 8

		
		features[9] =  -1 #favicondomain(wholeurl,domain,page))#feature 10 favicon domain
		#features.append(1 if "pat" in matches else -1) # have @
		features[10] = -1 if portnum == 80 else 1#feature 11
		features[11] = 1 if wholeurl.find('https')>6 else -1 #feature 12
		features[12] = -1 if domain in whitelistdomain else 1 # domain in whitelist
		#matches = None
		#domainmatches = None
		
		
		surl = None
		wholeurl = None
		#URL section end

		if html:
			#HTML section Start
			
			features[13] = anchorURL(html) #feature 14 1.2.2. anchor
			features[14] = MSLtags(html, domain) # links in meta, script and link tags
			formhandle = findall('<form.*(?!>)action="([^ >"]*)".*>',html)
			features[15] = SFH(formhandle)
			features[16] = Formmail(formhandle)
			formhandle = None
			features[2] =  urlredirect(wholeurl,page)# feature 3 URL redirection
			features[18] = -1  #1.3.1. Website Forwarding this feature is kind of duplicate with feature 3
			features[19] = onmouseover(html)#features 20: onmouseover
			features[20] = disableightclick(html) #features 21: whether has "event.button==2" in source code
			features[21] = 1 if "window.open(" in html else -1 #features 22: whether has popup window
			features[22] = 1 if "<iframe" in html else -1 #features 23:whether use iframe
			#print(html)
			html = None
			page = None
		else:
			features[13] = -1
			features[14] = -1
			
			features[15] = -1
			features[16] = -1
			features[2] =  -1
			features[18] = -1  
			features[19] = -1
			features[20] = -1
			features[21] = -1
			features[22] = -1

		#HTML section End
		
		#DNS section
		splitURLSection = domain.split('.')
		try:
			if(splitURLSection > 2):
				url = splitURLSection[-2] + '.' + splitURLSection[-1]
			w = pythonwhois.get_whois(url)
		except Exception as e:
			w = None
		if w and 'status' in w:
			today = datetime.today()
			features[8] = f9whois(w,today)#feature 9, use whois system call to decide reg length
			features[17] = identity(w,splitURLSection[-2])#feature 18, whether url in the name
			features[23] = f24whois(w,today) #feature 24, creation date more than 6 months. 
			features[24] = -1 # have DNS record so -1
		else:
			features[8] = 0#unkown time
			features[23] = 0#unkown time
			features[17] = 1#url not in name
			features[24] = 1 #no DNS record
		today = None
		w = None
		#DNS section over


		
		
		#these features are too expensive to process
		features[25] = -1
		features[26] = 0
		features[27] = 1
		features[28] = -1
		features[29] = 0
		
		#print domainmat
		#print(features)
	except Exception as e:
		print(e)	
#pass		
	#print(features)
	return features

#a New thread for url
def urlthread(url):
	try:
		start = time.time()
		val = NNclassifier(urlfeatureextractor(url))
		#print(NNclassifier(urlfeatureextractor(url)), file = stdout)
		end = time.time()
		print(str(val) + "," + str((end-start)))

	except Exception as e:
		print(e)
	return
#main function is a infinite loop which will constant read stdin for every 1024 bytes and output the phishing classification of all the urls in that list.
if __name__ == "__main__":
	#count = 0
	try:
		while True:
		
			try:
				
				urllist = findall(URL_REGEX,stdin.readline())#This regex will give us a turple with an URL followed by a few None string.
			#with ThreadPoolExecutor(2) as executor:
				if not urllist:
					continue
			except Exception as e:
				continue
			for url in urllist:
				if not url[0]:
					continue
				url = url[0]
				#print(url)
				#executor.submit(urlthread,url)
				#print("COUNT: " + str(count) + " URL:" + str(url))
				Thread(None,urlthread,None,(url,)).start()
				#count = count + 1
				#urlthread(url)
	except Exception as errtxt:
		print(errtxt)
	#	pass

