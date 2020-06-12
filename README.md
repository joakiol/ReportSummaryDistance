# ReportSummaryDistance
Code repository for TMA4500 Industriell matematikk, fordjupingsprosjekt. 

This code creates document embeddings for documents, and calculates document distance between them. 
The objective is to measure quality of summaries for reports. 

`main.py` contains configurations to run the plots in the report. Everything can be run from the main. 
Some packages must be installed. Apart from the usual stuff, this mainly involves gensim, nltk and sentence-bert. 
A few elements from nltk must be downloaded manually. This is done by adding the line `nltk.download('punkt')` in the main (for example if punkt is missing). 

`data.py` contains data objects and dataloaders. 

`analysis.py` contains the functionality for analysing the data. 

`TFIDFModel`, `LSAModel`, `LDAModel`, `Word2vecModel`, `Doc2vecModel` and `BertModel` contains the models. Implementations from gensim and sentence-bert are used. These files contain funtionality to make them easier to work with. 

The `data` folder contains the data (except the Vendu data, which will not be uploaded). This includes the SciSummNet-2019 data, the Concept-Project Matching data and the ACL Anthology Network corpus. The data is structured as one file per document, with one sentence per line in the files. Reports and summaries are treated as independent documents. 
