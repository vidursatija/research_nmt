# Research on Neural Machine Translation
This code base is a research conducted on RNNs to improve training speed and accuracy.

**Status: Failed**


## Intuition
It occured to me that the translator model shouldn't be very different from the normal RNN prediction model.
1 single RNN should act as an encoder as well as a decoder.
So, if we have 2 RNNs we can insert a neural network between them to create a translator model.
This new translator model should certainly reduce the time involved in training multiple seq2seq models as the model would be half trained.


## Experiment
1. I trained the simplest sequence2sequence model to convert words to their sounds (Alphabets to phonemes).

2. Next I trained a pure alphabets RNN in a seq2seq way i.e I trained a seq2seq model to produce the same alphabets using only 1 shared RNN.
Model: __alphabets -> RNN_alpha -> final_state -> tanh(matmul(weights, final_state) + bias) -> RNN_alpha -> logits__
Similarly this was followed for phonemes.

3. Now for the final translator model, I used the following model:
__alphabets -> RNN_alpha -> final_state -> tanh(matmul(new_w, final_state) + new_b) -> RNN_phoneme -> logits__


### Results
|Model | Log Perplexity|
|------|---------------:|
|Alphabets| 0.1 |
|Phonemes | 0.22 |
|End2End trained| *2* |
|New custom model| *13* |

#### Conclusion
1. We can certainly conclude that we can train 1 RNN to be used as an encoder and decoder.
2. Translation isn't quite efficient.
