
PreRequisites :
	Ubuntu 20.04 LTS
	Python 3.8.10
	nltk and numpy libraries, these are installed in make command if not already there.

Running assignment requires following steps:

0) Unzip the 20111038-ir-systems.zip and 20111038-qrels.zip
1) first place the unzipped english-corpora in 20111038-ir-systems folder
2) place the queries.txt in 20111038-ir-system folder (queries.txt is the input file available in 20111038-qrel.zip folder)
3) run make clean 
4) run make
5) question1 will start preprocessing, it takes 5-8 minutes (NOTE: The files from english-corpora are replaced with tokenised and stemmed data)
6) question2 will start with building posting lists , after 2 mins - it prompts for the input file name (NOTE: give it with extension example queries.txt)
7) the output will be in QRels format for 3 different model, top 10 document-ids with extension(because docid specified as filename in assignment.pdf) is generated.



######################################################################################################################################################################
The 20111038-qrels.zip contains QRels.txt which is the ground truth and answer for Q3 containing 10 relevant documents ranked in top 10 order (top to bottom),
and a queries.txt file which is the input file mentioned in Q5 contains queryid and tab-space and the query string.


In Q1 - Preprocessing stem , Nltk library is used along with PorterStemmer and word tokenizer - text is preprocessed for punctuations, upper case, non ascii characters(removed) and stop words are removed. The javascript elements are not considered as stopwords because in huge set of documents they will get normalised and does not hamper the scores.
The numbers are not removed considering, the dates and values add to the weightage of the data some times when N gram tokeniser is used.

The queries also go through the above preprocessing steps before evaluation.

In Q2 - Boolean Model - Though i have implemented the and/or/not logic, later as sir suggested - only 'and' was considered.
tfidf model - normalised tfidf model was used to mitigate the variations in relevance factors based on the frequency in different sized documents.
BM25- The full variant of bm25 is used, and the hyperparameters of k1,k2,b were tested for different results and the one's which provided best optimal results were considered.
(The answers might vary based on workstations of Windows and Linux as the libraries and precision in division varies)

in tfidf a runtime warning arises due to nan values arised in div by zero which is handled - this is beacuse of normalisations few values go extremely small, So ignore that.


The output of Q4 is provided in QRels format provided and as suggested by sir as qrel-boolean.txt, qrel-tfidf.txt, qrel-bm25.txt. Top 10 documents given, you can consider top 5


Overall assignment will not take more than 20 minutes tops to run.


I have referred the articles like medium,towardsdatascience and kaggle for concept building besides from Sir's Lectures.


If Any Doubt or Discrepancies is raised, feel free to contact me in leoevenss20@iitk.ac.in

