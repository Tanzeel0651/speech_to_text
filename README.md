# speech_to_text
Record your own audio and train your model to recognize only your voice.

STEP 1: Record audio
Write all the words to be recorded in transcribe.txt .
Run "record_audio.py" the words from text file will pop and you can speak to record file.
Recorded audio will be saved in audio/

STEP 2: Denoise
Run "remove_noise.py" from the same directory this will remove noise from every recorded audio file.

STEP 3: Train
Run "seq2seq.py this will train the model on your audio and would save in model/

Classic Encoder Decoder Model

convolution layer
Convolution layer
Max Pooling
Bidirectional LSTM
LSTM
Dense
Dense

optimizer = 'rmsprop'
loss = 'categoriacal crossentropy'
