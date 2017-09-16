# Research on Universal Encoder-Decoder for seq2seq models
This code base is a research conducted on RNNs to improve training speed and accuracy.

**Status: Failed**


## Intuition
It occured to me that the translator model shouldn't be very different from the normal RNN prediction model.
So, if we have 2 RNNs we can insert a neural network between them to create a translator model.
This new translator model should certainly reduce the time involved in training multiple seq2seq models as the model would be half trained.


## Experiment
1. I trained the simplest seq2seq model to convert words to their sounds (Alphabets to phonemes).

2. Next I trained a pure alphabets RNN in a seq2seq way i.e I trained a seq2seq model to produce the same alphabets using only 1 shared RNN.
Model: __alphabets -> RNN_alpha_en -> final_state -> matmul(V, tanh(matmul(weights, final_state) + bias)) + V_bias -> RNN_alpha_de -> logits__
Similarly this was followed for phonemes.

3. Now for the final translator model, I used the following model:
__alphabets -> RNN_alpha -> final_state -> matmul(V_new, tanh(matmul(new_w, final_state) + new_b)) + V_bias_new -> RNN_phoneme -> logits__


### Results
|Model | Log Perplexity|
|------|---------------:|
|Alphabets| 1.51 |
|Phonemes | 1.02 |
|End2End trained| *6* |
|New custom model| *13* |

