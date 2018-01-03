
Pre-requisite:
nltk (TweetTokenizer)
Keras
Tensorflow
numpy
scipy
gensim (if you are using word2vec)
itertools

To train the model(using Tensorflow):

git clone git@github.com:AniSkywalker/SarcasmDetection.git

cd SarcasmDetection/src/

You can find the trained model file in the following link
https://drive.google.com/drive/folders/0B7C_0ZfEBcpRbDZKelBZTFFsV0E?usp=sharing

Download the trained model in /resource/text_model/weights/

run the script
python sarcasm_detection_model_CNN_LSTM_DNN.py

If you want to train the model with your own data, you can place your the Train, Development and Test data file at /resource/train, /resource/dev, /resource/test folder correspondingly.
The system accept dataset in the following format:
id\<tab\>label\<tab\>tweet (see train file for example)

Please cite the following paper

<b>Fracking Sarcasm using Neural Network.</b>
Aniruddha Ghosh and Tony Veale. 
7th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (WASSA 2016). 
NAACL-HLT. 16th June 2016, San Diego, California, U.S.A. 

===============================================================================================

To run the model with context information and psychological dimensions(Using Tensorflow):

python sarcasm_context_moods.py

Please cite the following paper

<b>Magnets for Sarcasm: Making Sarcasm Detection Timely, Contextual and Very Personal</b>
Aniruddha Ghosh and Tony Veale
Conference on Empirical Methods in Natural Language Processing (EMNLP).
7th-11th September, 2017, Copenhagen, Denmark.

Sample of train, dev, and test files are added for both versions.

For test data set, please contact at aniruddha.ghosh@ucdconnect.ie
