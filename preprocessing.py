# Import Module
import os
import re
import string
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
import nltk
nltk.download('stopwords') # download if not present
nltk.download('punkt')


# Folder Path
path=os.getcwd()
path=path+f"/english-corpora" # changing path
os.chdir(path)

# Read text File


def read_text_file(file_path,i):
    fname=Path(file_path).stem # used to remove the extension
    with open(file_path, 'r',encoding='utf-8') as f:
        
        text=clean_text(f.read())
        f.close()
        os.remove(file_path) #replacing existing files
        output(text,fname)
    

        
def clean_text(text):
    text = ''.join(' ' if c in string.punctuation else c for c in text) # removing punctuations
    encoded_string = text.encode("ascii", "ignore") # removing non ascii characters
    
    decode_string = encoded_string.decode()
    return decode_string
    

        

    
def stemming(text):
    ps =  PorterStemmer() #using porter stemmer


    text1=""
    words = word_tokenize(text) #doing word tokenisation
    for w in words:
        text1+=ps.stem(w)+" "
    return text1

    
    
def output(text,i):
    text1=""
    
    words = text.split()
    
    filename=i+".txt"
    with open(filename, 'a', encoding='utf8') as f:
        
        stop_words = set(stopwords.words('english')) #removing stop words
        
        for r in words:
            r=r.lower() # casting uppercase to lowercase
            if not r in stop_words:
                
                text1+=r+" "
            
        

        
        text=stemming(text1)
        f.write(text)
    
    
        f.close()

i=0
# iterate through all file
print("started pre-processing")
    
for file in os.listdir():
	
    i+=1
    if file.endswith(".txt"):
       
        file_path = os.path.join(path,file);

        # call read text file function
        read_text_file(file_path,i)
print("done pre-processing")





