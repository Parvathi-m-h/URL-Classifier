from sklearn.feature_selection import mutual_info_classif
import csv
import numpy as np
import random
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
import csv
import copy
import sys
from sklearn.metrics import precision_recall_fscore_support
from csv import reader
from urlparse import urlparse
import urllib2
from xml.dom import minidom
import re
import pygeoip
import urllib


not_found=0

def find_rank_country(domain,ele,attribute):
    for subelement in domain.getElementsByTagName(ele):
    	#print "subelement",subelement
        if subelement.hasAttribute(attribute):
            return subelement.attributes[attribute].value
    return not_found
        

def alexa_ranking(host):

        xml_path='http://data.alexa.com/data?cli=10&dat=snbamz&url='+host
        #print "xml_path",xml_path
        try:
            xml= urllib2.urlopen(xml_path)
            #print "xml"
            domain =minidom.parse(xml)
            #print "domain, ",domain
            host_rank=find_rank_country(domain,'REACH','RANK')
            
            country_rank=find_rank_country(domain,'COUNTRY','RANK')
            return host_rank,country_rank

        except:
            return not_found,not_found



def get_tokens(url_part):

	if url_part=='':
            return 0,0,0
	tokens=re.split('\W+',url_part)
	# print tokens
	tokens=list(tokens)
	num_tokens=0
	largest=0
	total_len=0
	for token in tokens:
		# print "token:,",token
		l=len(token)
		total_len+=l
		# print "total_len",total_len
		if l>0:
			# print "num_token",num_tokens
			num_tokens+=1
			# print "num_token2",num_tokens
		if largest<l:
			largest=l
			# print "largest",largest
	try:
		avg_token_len=((total_len)/(num_tokens*1.0))
		return avg_token_len,num_tokens,largest
	except:
		return 0,num_tokens,largest


def security_sensitive_words(tokens):
	security_sensitive_words=['confirm', 'account', 'banking', 'secure', 'ebayisapi', 'webscr', 'login', 'signin','auth']
	count=0
	for words in security_sensitive_words:
		if(words in tokens):
			count+=1;
	return count

def get_asn(host):
	try:
		g = pygeoip.GeoIP('GeoIPASNum.dat')
		asn_string=g.org_by_name(host)
		# print "asn",asn_string.split()[0]
		asn=int(asn_string.split()[0][2:])
		return asn
	except:
		return  not_found

def spl_char(url):
	string=""
	res=string.join(e for e in url if not e.isalnum())
	return len(res)	

def find_unicode(url):
	count=0
	res=0
	for e in url:
		if isinstance(e,unicode):
			count+=1
		
	if count > 0:
		res=1
	return res


def safebrowsing(url):
    
    
    name = "api"
    ver = "1.5.2"

    req = {}
    req["client"] = name
    req["key"] = api_key // get your own api key
    req["appver"] = ver
    req["pver"] = "3.0"
    req["url"] = url 

  

    params = urllib.urlencode(req)
    try:
		req_url = "https://sb-ssl.google.com/safebrowsing/api/lookup?"+params
		res = urllib2.urlopen(req_url)

		if res.code==204:
			return 1
		elif res.code==200:
			return 0
    except:
    	return not_found
	


def readTestData():

	# ds = open(sys.argv[2])
	# readData=csv.reader(ds)
	# testData=list(readData)
	# testData=np.array(testData)
	features=list()
	url=sys.argv[2]
	tokens=re.split('\W+',url)
	if url.find("http:/") == -1:
		new_url="http://"+url
	else:
		new_url=url
	url_parts=urlparse(new_url)
	features.append(new_url)#0="URL"
	# url_parts=urlparse(row[0])
	host=url_parts.netloc
	path=url_parts.path
	# print "host",host
	# print "path",path

	features.append(host)
	
	features.append(path)#path
	host_rank,country_rank=alexa_ranking(host)
	features.append(host_rank)
	features.append(country_rank)
	
	features.append(len(new_url))#'Length_of_url'
	features.append(len(host))#'Length_of_host'
	
	features.append(new_url.count('.'))
	features.append(new_url.count('/'))
	features.append(new_url.count('-'))

	spl_count=spl_char(new_url)
	features.append(spl_count)

	avg_token_len,num_tokens,largest_token=get_tokens(new_url)
	features.append(avg_token_len)
	features.append(num_tokens)
	features.append(largest_token)

	avg_domain_len,domain_num_tokens,domain_largest_token=get_tokens(host)
	# print avg_domain_len,domain_num_tokens,domain_largest_token
	features.append(avg_domain_len)
	features.append(domain_num_tokens)
	features.append(domain_largest_token)

	avg_path_len,path_num_tokens,path_largest_token=get_tokens(path)
	features.append(avg_path_len)
	features.append(path_num_tokens)
	features.append(path_largest_token)

	sec_sens_count=security_sensitive_words(tokens)
	features.append(sec_sens_count)
	
	asn=get_asn(host)
	features.append(asn)

	unicode_found=find_unicode(new_url)
	features.append(unicode_found)

	safe_res=safebrowsing(new_url)
	features.append(safe_res)
	
	return features

class url_svm(object):
	def __init__(self):
		# self.split_ratio = 0.7
		ds = open(sys.argv[1])
		rdr = csv.reader(ds)
		self.data = list(rdr)
		self.data = random.sample(self.data, len(self.data))
		self.data = np.array(self.data)
		ds.close()

	def split_classLabel(self):
		cols = np.shape(self.data)[1] #returns no of rows and columns as a tuple
		self.X = self.data[:,:cols-1]
		self.X=self.X[:,3:]
		# print self.X
		self.X = self.X.astype(np.float)
		self.y = self.data[:,cols-1]
		self.y = np.array(self.y)
		#self.y = self.y.astype(np.int)
		y = np.ravel(self.y,order='C') #flattens the array

	def SVM(self):
	

		model=svm.LinearSVC(loss='hinge', intercept_scaling=1000)
		
		testData=readTestData()
		# cols = np.shape(testData)[1] #returns no of rows and columns as a tuple
		print testData
		testData=np.array(testData)
		X=testData[3:]
		# print self.X
		X =X.astype(np.float)
		model.fit(self.X,self.y)
		X=X.reshape(1,-1)
		result=model.predict(X)
		print result




u = url_svm()
u.split_classLabel()

u.SVM()
