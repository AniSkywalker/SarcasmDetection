# SarcamDetection
Sarcasm detection on tweets using neural network.
[This repository] perform[s] semantic  modelling  of  sentences  using  neural  networks for the task of sarcasm detection ([Ghosh & Veal, 2016](http://www.aclweb.org/anthology/W16-0425)). 
## Pre-requisite
- nltk (TweetTokenizer)
- Keras
- Tensorflow
- numpy
- scipy
- gensim (if you are using word2vec)
- itertools

## Cloning the repository
```
git clone git@github.com:AniSkywalker/SarcasmDetection.git
cd SarcasmDetection/src/
```
If you want to use the pre-trained model, you'll have to [download it](https://drive.google.com/drive/folders/0B7C_0ZfEBcpRbDZKelBZTFFsV0E?usp=sharing) from Google Drive and save it into `/resource/text_model/weights/`.

## Using this package
This code is run by the following command:
```
python sarcasm_detection_model_CNN_LSTM_DNN.py
```

##### Citation
Please cite the following paper when using this code:

> **Fracking Sarcasm using Neural Network.**
> Aniruddha Ghosh and Tony Veale. 7th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (WASSA 2016). NAACL-HLT. 16th June 2016, San Diego, California, U.S.A. 

## Output
The supplied input is rated as either **0** meaning _non-sarcastic_ or **1** meaning _sarcastic_.

## Training
If you want to train the model with your own data, you can save your _train_, _development_ and _test_ data into the `/resource/train`, `/resource/dev` and `/resource/test` folders correspondingly.

The system accepts dataset in the tab separated format — as shown below. An example can be found in [`/resource/train/train_v1.txt`](https://github.com/AniSkywalker/SarcasmDetection/tree/master/resource/train). 
```
id<tab>label<tab>tweet
```

## Context information
To run the model with context information and psychological dimensions (using Tensorflow) run:
```
python sarcasm_context_moods.py
```

##### Citation
Please cite the following paper when using context information and psychological dimensions:
> **Magnets for Sarcasm: Making Sarcasm Detection Timely, Contextual and Very Personal**
> Aniruddha Ghosh and Tony Veale. Conference on Empirical Methods in Natural Language Processing (EMNLP). 7th-11th September, 2017, Copenhagen, Denmark.

## Notes
- Samples of _train_, _dev_, and _test_ files are included for both versions.
- For a test data set, please contact at aniruddha.ghosh@ucdconnect.ie
