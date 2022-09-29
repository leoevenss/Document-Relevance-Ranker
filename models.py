
import glob
import os
from pathlib import Path
from math import log
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import collections
import nltk
import math
import numpy as np
import operator
import string

idx=dict() # inverted index 
dlt=dict() # document length table
avgdlt=0  # average doc length
_stemmer = nltk.stem.porter.PorterStemmer()

path=os.getcwd()
os.chdir(path)



def read_corpus(): # reading corpus and building required data structure
    pat=path+f"/english-corpora/"

    file_folder = pat
    os.chdir(file_folder)
    for file in os.listdir():
    	
    	fname = file
    	doc_id=Path(fname).stem  #to get the name of doc
    	file = open(file , "r", encoding='utf-8')
    	text = file.read()
    	
    	words = word_tokenize(text)
    	inverted_index(words,doc_id)
    	doc_len_table(doc_id,len(words))
    	avgdlt=get_avg_doc_len()



        
#### Building  document table ##### 

def doc_len_table(doc_id,length): # building document length table - docid: length
    dlt[doc_id]=length

def get_length(doc_id): # get length of each document
    if doc_id in dlt:
        return dlt[doc_id]
    else:
        print('%s not found in table' % str(doc_id))

def get_avg_doc_len(): # average document length
        sum=0
        for length in dlt.values():
            sum+=length
        return float(sum) / float(len(dlt))


#### Building inverted index ####

def inverted_index(words,doc_id): # building inverted index :- word : {docid1: frequency, docid2:frequency, ... }
    for word in words:
        if word in  idx:
            if doc_id in idx[word]:
                idx[word][doc_id]+=1
            else:
                idx[word][doc_id]=1
        else:
            fd=dict()
            fd[doc_id]=1
            idx[word]=fd


def get_posting_list(term):
    if term in idx:
        return idx[term].keys()
    else:
        return list()

##################################################### Boolean retrieval system ###############################################        


def _boolean_parse_query(inf_tokens): #### only and operation is activated as Sir told it is sufficient; if or and not are needed, add extra line during stop word removal 
        prec = {}
        prec['NOT'] = 3
        prec['not']=3
        prec['AND'] = 2
        prec['and'] = 2
        prec['or'] = 1 ##using precedence to different operations
        prec['OR'] = 1
        prec['('] = 0
        prec[')'] = 0    

        out = []
        oper_stack = []

        for token in inf_tokens:
            if (token == '('):
                oper_stack.append(token)
            
            # operating for brackets, if ) pop all from stack until ( is encountered
            # if operator, pop from stack to queue if they are of higher precedence
            
            elif (token == ')'):
                oper = oper_stack.pop() #doing a posftfix operation
                while oper != '(':
                    out.append(oper)
                    oper = oper_stack.pop()
            
            
            elif (token in prec):
            
                if (oper_stack):
                    cur_oper = oper_stack[-1]
                    while (oper_stack and prec[cur_oper] > prec[token]):
                        out.append(oper_stack.pop())
                        if (oper_stack):
                            cur_oper = oper_stack[-1]
                oper_stack.append(token)
            else:
                out.append(token.lower())

        # if there are operators on the stack, popping them into the output queue
        while (oper_stack):
            out.append(oper_stack.pop())
        
        return out

    

def and_operation(left_list, right_list): # doing intersection of two posting lists
    if len(left_list)>=len(right_list):
        temp = set(left_list)
        result = [value for value in right_list if value in temp]
    else:
        temp = set(right_list)
        result = [value for value in left_list if value in temp]
    return result

def or_operation(left_list,right_list): # doing union of two posting lists
    result =list(set(left_list) | set(right_list))
    return result

def not_operation(term_list): #doing negation of the universal set
    result=list()
    for i in dlt.keys():
        if i in term_list:
            pass
        else:
            result.append(i)
    return result

def boolean_process_query(query): ### pass query as it is in boolean #just dont remove the stop words
        results_stack = []
    
        
        length=len(query)
        newl=list()
        for i in range(0,length-1):
            newl.append(query[i])
            if(query[i] not in ('AND','and','OR','or','NOT','not') and query[i+1] not in ('AND','and','OR','or','not','NOT')): #adding and in between
                newl.append('and')

        newl.append(query[len(query)-1])
        
        postfix_queue = collections.deque(_boolean_parse_query(newl)) # get query in postfix notation as a queue
        
        while postfix_queue:
                token = postfix_queue.popleft()
                result = []
            

                if (token != 'AND' and token !='and' and token !='or' and token != 'OR' and token != 'NOT' and token!='not'):
                    token = _stemmer.stem(token) # stem the token
                
                    if (token in idx):
                        result = get_posting_list(token)
                    else:
                        result=list()
                elif (token == 'AND' or token =='and'):
                    right_operand = results_stack.pop()
                    left_operand = results_stack.pop()
                    result = and_operation(left_operand, right_operand)   # evaluate AND

                elif (token == 'OR' or token=='or'):
                    right_operand = results_stack.pop()
                    left_operand = results_stack.pop()
                    result = or_operation(left_operand, right_operand)    # evaluate OR

                elif (token == 'NOT' or  token=='not'):
                    right_operand = results_stack.pop()
                    result = not_operation(right_operand) # evaluate NOT

                results_stack.append(result)                        
        return results_stack.pop()

        
###################################################### Computing tfidf function ###############################################         





doc_idf=dict()
doc_tf_norm=dict()
query_tf=dict()
query_idf=dict()

def doc_normalised_tf():
    for term in idx.keys():
        if term not in doc_tf_norm:
            d=dict()
            doc_tf_norm[term]=d
        for doc in idx[term]:
        
            doc_tf_norm[term][doc]=idx[term][doc]/float(dlt[doc])

    
    
def _doc_idf():
    for term in idx.keys():
        #print(idx[term])
        doc_idf[term]=1+math.log(float(len(dlt))/len(idx[term]))

        

def _query_tf(query):
    
    ### pass stemmed and tokenised query
     
    for term in query:
        #normalizedquery = query.lower().split()
        query_tf[term]=query.count(term.lower())# / float(len(query))
    
def _query_idf(query):
    ## pass stemmed and tokenised query
    
    for term in  query:
        if term in doc_idf.keys():
            query_idf[term]=doc_idf[term]
        else:
            query_idf[term]=1.0

            
            
            
def cosine_similarity(query,doc):
    dot_prod=0
    den_query=0
    den_doc=0
    query_tfidf=dict()
    doc_tfidf=dict()
    for term in query:
        query_tfidf[term]=query_tf[term]*query_idf[term]
        doc_tfidf[term]=0
        if term in doc_tf_norm:
            doc_list= doc_tf_norm[term].keys()
            if doc in doc_list:
                doc_tfidf[term]=doc_tf_norm[term][doc]*doc_idf[term]
        dot_prod+=query_tfidf[term]*doc_tfidf[term]
        den_query+=query_tfidf[term]*query_tfidf[term]
        den_doc+=doc_tfidf[term]*doc_tfidf[term]
    den_query=np.sqrt(den_query)
    den_doc=np.sqrt(den_doc)
    denom=den_query*den_doc
    cosine_sim=dot_prod/denom
    if math.isnan(cosine_sim):
        cosine_sim=0.0
    return cosine_sim
########################################################### Computing BM25 scores #############################################
k1 = 1.2 #hyperparameters
k2 = 100
b = 0.75
R = 0.0

def score_BM25(n, f, qf, r, N, dl, avdl): #computing bm25 scores
    K = k1 * ((1-b) + b * (float(dl)/float(avdl)) )
    f = log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
    s = ((k1 + 1) * f) / (K + f)
    t = ((k2+1) * qf) / (k2 + qf)
    return f * s * t


def run(query):
        results = []
        #for query in queries:
         #   print(query)
        results.append(run_query(query))
        return results

def run_query(query):
        query_result = dict()
        for term in query:
            if term in idx:
                doc_dict = idx[term] # retrieve index entry
                for docid, freq in doc_dict.items(): #for each document and its word frequency
                    score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(dlt),
                                       dl=get_length(docid), avdl=get_avg_doc_len()) # calculate score
                    if docid in query_result: #this document has already been scored once
                        query_result[docid] += score
                    else:
                        query_result[docid] = score
        return query_result


################################################ Query Processing ###########################################################
############################################################################################################################    


def process_queries(filename):
    
    stop_words = set(stopwords.words('english'))    
    
    #file name path
    queries=list()

    with open(filename) as f:
        
        lines=f.readlines()
        
        for line in lines:
            dummy=list()
            line=line.strip()
            line=line.split('\t')
            dummy.append(line[0].strip())
            
            text = ''.join(' ' if c in string.punctuation else c for c in line[1])
            encoded_string = text.encode("ascii", "ignore")
            decode_string = encoded_string.decode()
            text=decode_string.split()
        
            text1=''
            for r in text:
                r=r.lower()
                if not r in stop_words:
                    text1+=r+" "
            ps =  PorterStemmer()

            words = word_tokenize(text1)
            text1=''
            for w in words:
                text1+=ps.stem(w)+" "
        
    
            text1=text1.split()
            dummy.append(text1)
            queries.append(dummy)
    
    return queries

###############################################################################################################################    
def main():
	print("Building Posting Lists")
	read_corpus()
	doc_normalised_tf()
	_doc_idf()
    
	print("done building posting lists")
	os.chdir('..') # going one step previous for parent dir
	filename=input("Input filename with extension ")
	
	print("preprocessing query")
	querylist=process_queries(filename)
	#print(querylist)
	print("queries preprocessing done")
    
    ######boolean###############
	print("started boolean model")
	qid=0
	itera=1
	rel=1
	with open("qrels-boolean.txt", "w") as file1:
		for query in querylist:
			res=list()
			qid=query[0]
			res=boolean_process_query(query[1])
			j=0
			for r in res:
				j+=1
    			
				file1.write("{},{},{},{}\n".format(qid,itera,r+'.txt',rel))#getting 10
				if j>=10:
					break
        
	print("boolean model done")
    ### tfidf#######
	qid=0
	itera=1
	rel=1
	print("#################################################################################")
	print("starting tfidf model")
	with open("qrels-tfidf.txt", "w") as file2:
		
		for query in querylist:
			qid=query[0]
			_query_tf(query[1])
			_query_idf(query[1])
			scores={}
			docs=dlt.keys()
			for doc in docs:
				scores[doc]=cosine_similarity(query[1],doc)
			query_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)# sorting for top10
			res=query_scores[:10]
			for r in res:
				file2.write("{},{},{},{}\n".format(qid,itera,r[0]+'.txt',rel))
	    
				
           
	print("tfidf model done")
     
	print("#################################################################################")
	print("starting bm25 model")
     ########################## BM25 model ########################
	qid=0
	itera=1
	rel=1
	with open("qrels-bm25.txt", "w") as file3:
		for query in querylist:
			qid=query[0]
			results=run(query[1])
			for result in results:
				sscore = sorted(result.items(), key=operator.itemgetter(1)) #sorting for top 10
				sscore.reverse()
				for i in sscore[:10]:
					temp = (qid,itera,i[0],rel)
					file3.write("{},{},{},{}\n".format(temp[0],temp[1],temp[2]+'.txt',temp[3]))
	print("bm25 model done")
    ##################################################################################
    
    
        

if __name__ == '__main__':
    main()
    
    

