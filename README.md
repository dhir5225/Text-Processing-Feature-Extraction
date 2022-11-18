# Text-Processing-Feature-Extraction

### Step 1 : Data Preprocessing

### Pip install Spacy

### What’s spaCy?

spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in Python.

If you’re working with a lot of text, you’ll eventually want to know more about it. For example, what’s it about? What do the words mean in context? Who is doing what to whom? What companies and products are mentioned? Which texts are similar to each other?

spaCy is designed specifically for production use and helps you build applications that process and “understand” large volumes of text. It can be used to build information extraction or natural language understanding systems, or to pre-process text for deep learning.

### Features

The first thing you need to do in any NLP project is text preprocessing. Preprocessing input text simply means putting the data into a predictable and analyzable form. It’s a crucial step for building an amazing NLP application.

There are different ways to preprocess text: 

##### 1.Tokenization, 

##### 2.Stop word removal, 

##### 3.Stemming Or Lemmatization. 

Among these, the most important step is tokenization. It’s the process of breaking a stream of textual data into words, terms, sentences, symbols, or some other meaningful elements called tokens. A lot of open-source tools are available to perform the tokenization process.

### 1.Tokenization

DESCRIPTION-Segmenting text into words, punctuations marks etc.

Tokenization is the first step in any NLP pipeline. It has an important effect on the rest of your pipeline. A tokenizer breaks unstructured data and natural language text into chunks of information that can be considered as discrete elements. The token occurrences in a document can be used directly as a vector representing that document. 

Tokenization can separate sentences, words, characters, or subwords. When we split the text into sentences, we call it sentence tokenization. For words, we call it word tokenization.

Example of word tokenization

![image](https://user-images.githubusercontent.com/109084435/202626305-fc090d65-b052-473a-8b1c-936b31a60d02.png)

Example of sentence tokenization

![image](https://user-images.githubusercontent.com/109084435/202626461-6a9deaf5-b9a0-45a3-9db7-d46cf02ddd67.png)

#### Different tools for tokenization

#### 1.NLTK Word Tokenize

NLTK (Natural Language Toolkit) is an open-source Python library for Natural Language Processing. It has easy-to-use interfaces for over 50 corpora and lexical resources such as WordNet, along with a set of text processing libraries for classification, tokenization, stemming, and tagging.

You can easily tokenize the sentences and words of the text with the tokenize module of NLTK.

#### 2.Gensim word tokenizer

Gensim is a Python library for topic modeling, document indexing, and similarity retrieval with large corpora. The target audience is the natural language processing (NLP) and information retrieval (IR) community. It offers utility functions for tokenization

![image](https://user-images.githubusercontent.com/109084435/202626821-ad4706cd-822b-420f-b9e9-88362dcf4ae1.png)

#### 3.Tokenization with Keras

Keras open-source library is one of the most reliable deep learning frameworks. To perform tokenization we use: text_to_word_sequence method from the Class Keras.preprocessing.text class. The great thing about Keras is converting the alphabet in a lower case before tokenizing it, which can be quite a time-saver.

![image](https://user-images.githubusercontent.com/109084435/202626930-2cd03c8b-872a-4ecf-ad2d-9b7f59f527ce.png)

### 2.Stopword Removal:

Stopwords are the words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. For example, if you see the below example we can see the stopwords are removed. Nltk (natural language tool kit) offers functions like tokenize and stopwords. You can use the following template to remove stop words from your text.

##### Example of Stop Word Removal

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
input_text = “I am passing the input sentence here. so we will see what happens with this and.”
stop_words = set(stopwords.words(‘english’)) 
word_tokens = word_tokenize(input_text) 
output_text = [w for w in word_tokens if not w in stop_words] 
output = [] 
for w in word_tokens: 
    if w not in stop_words: 
        output.append(w) 
  
print(word_tokens) 
print(output)
#Printing Word Tokens and output(stop words removed)#
['I', 'am', 'passing', 'the', 'input', 'sentence', 'here', '.', 'so', 'we', 'will', 'see', 'what', 'happens', 'with', 'this', 'and', '.']
['I', 'passing', 'input', 'sentence', '.', 'see', 'happens', '.']

### 3.Stemming Or Lemmatization:

Stemming

Words are reduced to a root by removing inflection through dropping unnecessary characters, usually a suffix.It is the process of producing morphological variants of a root/base word.Stemming programs are commonly referred to as stemming algorithms or stemmers.

Often when searching text for a certain keyword, it helps if the search returns variations of the word. For instance, searching for “boat” might also return “boats” and “boating”. Here, “boat” would be the stem for [boat, boater, boating, boats].

Lemmatization

In contrast to stemming, lemmatization looks beyond word reduction and considers a language’s full vocabulary to apply a morphological analysis to words. The lemma of ‘was’ is ‘be’ and the lemma of ‘mice’ is ‘mouse’.

Lemmatization is typically seen as much more informative than simple stemming, which is why Spacy has opted to only have Lemmatization available instead of Stemming

Lemmatization looks at surrounding text to determine a given word’s part of speech, it does not categorize phrases.

##### Example:

For Stemming-

from nltk.stem import PorterStemmer
stemming=PorterStemmer()
words=["eating","eats","eaten","writing","writes","programming","programs","history","finally","finalized"]
for word in words:
    print(word+"---->"+stemming.stem(word))
eating---->eat
eats---->eat
eaten---->eaten
writing---->write
writes---->write
programming---->program
programs---->program
history---->histori
finally---->final
finalized---->final

stemming.stem('congratulations')
'congratul'

stemming.stem('sitting')
'sit'

For lemmatizer:

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
'''
POS- Noun-n
verb-v
adjective-a
adverb-r
'''
for word in words:
    print(word+"---->"+lemmatizer.lemmatize(word,pos='v'))
eating---->eat
eats---->eat
eaten---->eat
writing---->write
writes---->write
programming---->program
programs---->program
history---->history
finally---->finally
finalized---->finalize
lemmatizer.lemmatize("good",pos='v')
'good'

#### Note
Sentiment Analysis-- stemming
Chatbot---lemmatization

### Step 2: Feature Extraction

Feature Extraction is one of the trivial steps to be followed for a better understanding of the context of what we are dealing with. After the initial text is cleaned and normalized, we need to transform it into their features to be used for modeling. We use some particular method to assign weights to particular words within our document before modeling them. We go for numerical representation for individual words as it’s easy for the computer to process numbers, in such cases, we go for word embeddings.

#### Various methods of feature extraction and word embeddings practiced in Natural Language processing:

##### 1.Bag of Words:

The bag-of-words model is commonly used in methods of document classification where the (frequency of) occurrence of each word is used as a feature for training a classifier.

In this step we construct a vector, which would tell us whether a word in each sentence is a frequent word or not. If a word in a sentence is a frequent word, we set it as 1, else we set it as 0.
This can be implemented with the help of following code:

X = []
for data in dataset:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)
X = np.asarray(X)

##### 2.Term Frequency-Inverse Document Frequency (TF-IDF):

It can be defined as the calculation of how relevant a word in a series or corpus is to a text. The meaning increases proportionally to the number of times in the text a word appears but is compensated by the word frequency in the corpus (data-set).

##### Code-

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(2,3), max_features = 10)
tf_idf_matrix_n_gram_max_features = vectorizer_n_gram_max_features.fit_transform(book)
print(vectorizer_n_gram_max_features.get_feature_names())
print(tf_idf_matrix_n_gram_max_features.toarray())

##### 3.Word2Vec

Word2vec takes as its input a large corpus of text and produces a vector space with each unique word being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space. Word2Vec is very famous at capturing meaning and demonstrating it on tasks like calculating analogy questions of the form

For example, man is to woman as uncle is to ? (aunt) using a simple vector offset method based on cosine distance. For example, here are vector offsets for three word pairs illustrating the gender relation:

![image](https://user-images.githubusercontent.com/109084435/202632958-2bed8ac8-30db-4a61-8fa4-5121fe395144.png)

### Step 3: Choosing ML Algorithms

There are various approaches to building ML models for various text based applications depending on what is the problem space and data available.

Classical ML approaches like ‘Naive Bayes’ or ‘Support Vector Machines’ for spam filtering has been widely used. Deep learning techniques are giving better results for NLP problems like sentiment analysis and language translation. Deep learning models are very slow to train and it has been seen that for simple text classification problems classical ML approaches as well give similar results with quicker training time.


