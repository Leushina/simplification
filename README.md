# Text Simplification

Text Simplification (TS) is the process of modifying natural language to **reduce its complexity** and improve both readability and understandability. 
For different texts and goals, this may refer to different things, aim can be to simplify text for people who find it difficult to read long sentences or complex grammar (for example non-native speakers) or aim may reflect on the need to simplify technical factors for non-experts.

In this project different methods will be implemented, focusing on machine translation techniques.
Several models for Text Simplification are all Encoder-Decoder based. 
Two of them are **GRU-based** and one is **Transformer**. 

# Models
RNN-based models are using different pretrained word embeddings, i.e. Glove and Word2vec *(details for these models are in get_model.py)*.
Transformer uses its own embedding layers. *(see get_transformer.py)*
We will consider only English language. 

# Dataset
Training dataset (Newsela) were created by Wei Xu, Chris Callison-Burch, Courtney Napoles. 
It contains the data of 1130 news articles, where each article contains 5 versions (1 original and 4 simplified versions). 
The target audience for these articles is adult literacy learners (i.e., native speakers with poor reading skills), 
but the site creators suggest that the abridged articles can be used by instructors and learners of all ages.

# Evaluation
As the main metric for performance evaluation, we use **SARI**.
It compares simplified sentences not only with the initial sentence but also with simplified sentences available from the dataset (references). 
By doing so, this metric can positively score new words that are present in the reference sentence but not in an initial sentence.
In the same way, it can reward keeping necessary words and removing words present in initial sentences but omitted in reference.
 *(code for sari implementation taken from [here](https://github.com/XingxingZhang/pysari))*

The examples found and analyzed by us allow us to make a conclusion that model with Transformer architecture 
showing bestresult out of implemented models. 
Metrics are also confirm this fact,seeing as simplification (SARI) and readability (Flesch) are highest.

# API

![API example](https://drive.google.com/uc?id=1-YAJ_mvT6-Zfp6ETC4VQEfR_Z-L8lSQm )

Therefore, we created an API for simplification using Python and Flask framework. 
We built small inverted index for search from articles on [reuters.com](https://www.reuters.com) 
since training data is also consist of news articles. 
A request (sentence to be simplified, from chosen article or user's text) sent via API and preprocessed 
(punctuation signs separated with additional space, all letters brought to lowercase). 
Then loaded pretrained model is used for simplification and result of its work sent back to user. User can evaluate it,  modify it to better simplification and save it 
[here](https://docs.google.com/spreadsheets/d/1R-322yozb-mIijinjg1fHZCr49_nFP5p4zY-cPXt0XM/edit#gid=0) 
for future research.

