# Encoder-Decoder Grammar Correction Program

This program uses PyTorch to train sequence-to-sequence models that can correct standard English sentences with 1 missing grammatical particle (e.g. a, the, to).

## Approach
Sentences labelled as correct are preprocessed from [Lang-8 Learner Corpora](http://cl.naist.jp/nldata/lang-8/). Custom sentences may also be added from data/custom_sentences.txt to improve the variety of sentences. Some sentences will then perturbed by removing 1 randomly selected grammatical particle from data/particles.txt. 2 models will be trained with these data preprocessed data. The first model is an encoder-decoder that has an output size equivalent to the total number of unique words amongst all the sentences. This model is used to predict a word given the particular word in the sentence. The second model is another encoder-decoder that has an output size of 3. This model is used to tell the first model whether a particular word should be changed to the predicted word, remain, or be removed. The model can be implemented as a character/word level model.

## Datasets
- [Lang-8 Learner Corpora](http://cl.naist.jp/nldata/lang-8/)
- [Nucle 3.2](http://www.comp.nus.edu.sg/~nlp/conll14st.html) previously used
- [Atpaino Deep Text Corrector](https://github.com/atpaino/deep-text-corrector) previously used

## Descriptions
The program is fully interactive, therefore only 'main.py' needs to be run. 
-`main.py`: run this to to start the program, follow the ui's instructions to navigate between preprocessing and training/using the model. Use '--help' for more information about user arguments.
-`correct.py`: this file allows you to train/evaluate model, show model history, and correct a sentence.
-`preprocess.py`: this file allows you to preprocess data by number/proportion of samples.
-`data/`: this folder contains all the necessary data (some not uploaded due to copyright policies)
-`models/`: this folder contains model class implementations and saved model information
-`ui/`: this folder contains code for the ui

## Papers
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
- [Neural Network Translation Models for Grammatical Error Correction](https://arxiv.org/pdf/1606.00189.pdf)
