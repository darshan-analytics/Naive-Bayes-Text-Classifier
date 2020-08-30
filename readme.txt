Darshan Desai
B00816526
ddesai9@binghamton.edu

I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else. I understand that if I am involved in plagiarism or cheating I will have to sign an official form that I have cheated and that this form will be stored in my official university record. 
I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that I will receive a grade of “F” for the course for any additional offense.

-- Darshan Ramchandra Desai
Assignment No 2 - Naive Bayes Algorithm

Description: Used Naive Bayes Classifier for text classification.

Language: Python

Tested on: Windows, Linux

How to compile:
	Use the below command to run the program on command line/ linux terminal:

         python3 naive_bayes.py

Kindly Install all the libraries which I used -
import math
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

Also keep the train and test data folders in the same directories as program.
Ideally wihtout stopwords accuracy should be greater but for this particular dataset accuracy without stopwords is lower than accuracy with stopwords.
This might be due to less impact of stopwords in the data.