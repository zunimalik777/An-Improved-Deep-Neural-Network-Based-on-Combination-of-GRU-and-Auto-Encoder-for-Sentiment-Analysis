# An Improved Deep Neural Network Based on Combination of GRU and Auto Encoder for Sentiment Analysis

# Abstract 
Sentiment analysis is a particularly common task for determining user thoughts and has been widely used in Natural Language Processing (NLP) applications. Gated Recurrent Unit (GRU) was already effectively integrated into the NLP process with comparatively excellent results. GRU networks outperform traditional recurrent neural networks in sequential learning tasks and solve gradient vanishing and explosion limitation of RNNs. In this paper, a novel approach as known Normalize Auto-Encoded GRU (NAE-GRU) was proposed, in order to reduce dimensionality of data through an Auto-Encoder and enhance the performance of the proposed approach by using batch normalization. Empirically, we demonstrate that the proposed model, with minor hyperparameters modification, and statistic vectors optimization, achieves outstanding sentiment classification performance on benchmark datasets. The developed NAE-GRU approach outperforms than other different traditional methods in terms of accuracy and convergence rate. The experimental results have showed that the developed approach accomplished excellent performance than existing approaches on four benchmark datasets included, Amazon review, Yelp review, IMDB and SSTb. The experimental results have showed that the developed approach is proficient to reduce the loss function, and capture long-term relationships with an effective design that achieved excellent results as compared state-of-the-art methods. 

# Introduction
This tutorial introduces how to train  Normalize Auto-Encoded GRU (NAE-GRU) model and comparative approaches for sentiment analysis. This code is written in Python 3.7 by using gensim(https://radimrehurek.com/gensim/) including numpy, scipy, Sklearn, pandas, matplotlib libraries with Keras packages. Complete simulations were performed on Intel Core i7-3770CPU @3.40 GHz, and 8GB of RAM machine. The training simulated of the proposed and comparative models is executed 5 time for each combination of momentum with learning rate 0.001 and mini-batch size. To avoid the overfitting issue, we adapted the dropout technique, with a dropout rate of 0.2 for the GRU layer and 10−5 for the coefficient λr of L2 regularization. You can checkout github for more details.

# Preprocessing the Corpus
To train proposed and comparative approaches  with python libraries, you need to put each document into a line without punctuations. So, the output file should include all sentences and each sentence should be in a line. Moreover, Gensim library provides methods to do this preprocessing step. However, tokenize function is modified for sentence classification. You can run preprocess.py to modify your corpus.

# Training NAE-GRU Model
After preprocessing the dataset, training NAE-GRU model with gensim library is very easy. You can use the code below to create NAE-GRU model. 

# JavaScript
const OpenTC = require('opencc');
const converter = new OpenTC('uiuc.word');
converter.convertPromise("Text character").then(converted => {
  console.log(converted);  // text character
});

# Python
PyPI pip install opentc (Windows, Linux, Mac)
import opentc
converter = opentc.OpenTC('uiuc.word')
converter.convert('word')  #Text character
}

# Command Line
opencc --help
opencc_dict --help
opencc_phrase_extract --help

# Others (Unofficial)
Swift (iOS): SwiftyOpenTC
Java: opentc4j
Android: android-opentc
PHP: opentc4php
Pure JavaScript: opentc-js
WebAssembly: wasm-opentc
