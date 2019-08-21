import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, MaxPooling1D, GlobalMaxPool1D
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers import LSTM, Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import TimeDistributed, Dense
from keras.layers import RepeatVector, Bidirectional, GlobalMaxPooling1D
from keras.models import Model
from keras.layers import Flatten, ConvLSTM2D, Dropout
from keras.models import model_from_json


# intitalizing 
MAX_SEQ_LENGTH = 100
MAX_VOCAB_SIZE = 20000
BATCH_SIZE = 128
EPOCHS = 300
LATENT_DIM = 30
EMBEDDING_DIM = 100
LENGTH = 60
count = 10
path = 'audio/'
WORDS = []



def get_fft(file_path, sr, duration):
    fft_bank = []
    full_data, sr = librosa.load(file_path, sr=sr)
    audio_duration = librosa.get_duration(full_data, sr)
    dur = np.arange(0.0, audio_duration, duration)
    dur = np.append(dur, audio_duration)
    sample_fft = []
    print(file_path.split('/')[-1])
    for i in range(0, len(dur)-1):
        print(dur[i])
        data, sr = librosa.load(file_path,
                                offset=dur[i],
                                duration=dur[i+1],
                                sr=sr)
        sample_fft.append(np.array(fft(data)))
    fft_bank.append(np.array(sample_fft))
    return sample_fft

target_texts_inputs = []
target_texts = []
fft_bank = [0 for _ in range(count)]
for i,file in enumerate(sorted(os.listdir(path))):
    print(i, end='  ')
    fft_bank[i] = get_fft(path+file,
                             sr=16000,
                             duration=0.20)
    i += 1
    word = ' '.join(x for x in file.split('.')[0].lower())
    WORDS.append(file.split('.')[0].lower())
    target_texts_inputs.append('<sos> '+word)
    target_texts.append(word + ' <eos>')
    if (i==count): break    

# with open('trained_words.txt', 'w') as f:
#     for word in WORDS: f.write(word+'\n')
    
max_len_audio = 0
max_len_sample = 0
for sample in fft_bank:  
    if len(sample)>max_len_sample: max_len_sample=len(sample)
    for audio in sample:
        if audio.shape[0]>max_len_audio:
            max_len_audio = audio.shape[0]
print('Max Lenght Sample: ', max_len_sample)
print('Max Length Audio: ', max_len_audio)

####################
#max_len_sample = 6
#max_len_audio = 8000
####################
### PADDING FFT
#padded_fft = np.empty((count, max_len_sample, max_len_audio), dtype='float32')
padded_fft = []
for i, sample in enumerate(fft_bank):
    padded_sample_fft = []
    for audio in sample:
        padded_sample_fft.append(librosa.util.pad_center(audio, max_len_audio, axis=0))
        
    zeros = np.zeros((max_len_audio,), dtype='complex')
    for _ in range(max_len_sample-len(sample)):
        padded_sample_fft.append(zeros)
        
    padded_fft.append(np.array(padded_sample_fft))
        
padded_fft = np.array(padded_fft)
print('Padded FFT shape: ',padded_fft.shape)


tokenizer_outputs = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer_outputs.fit_on_texts(target_texts_inputs+target_texts)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

word2idx_output = tokenizer_outputs.word_index

num_words_output = len(word2idx_output) + 1
print('Num Word output: ',num_words_output)
max_len_target = max(len(s) for s in target_sequences)
print('Max length of target: ',max_len_target)

# Padding word sequences
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print("decoder_inputs[0]:", decoder_inputs[0])
print("decoder_inputs.shape:", decoder_inputs.shape)

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')
print("decoder_targets[0]:", decoder_targets[0])
print("decoder_targets.shape:", decoder_targets.shape)
print('Loading the pre-trained word vectors')


print('Loading pre trained word vectors')
 
word2vec = {}
with open('/home/tanzeel/Documents/glove_vector/glove.6B.%sd.txt' % EMBEDDING_DIM) as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.array(values[1:], dtype='float32')
        word2vec[word] = vec
         
print('Filling pre pre-trained embeddings')
num_words = min(MAX_VOCAB_SIZE, len(word2idx_output)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_output.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vec = word2idx_output.get(word)
    if embedding_vec is not None:
        embedding_matrix[i] = embedding_vec
         
# load pre-trained embeddings in embedding layer
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    #input_length=max_len_target
)
 
decoder_target_one_hot = np.zeros(
    (
        len(target_texts),
        max_len_target,
        num_words_output
    ), dtype='float32'
)

#assign the values
for i, t in enumerate(decoder_targets):
    for z, word in enumerate(t):
        decoder_target_one_hot[i, z, word] = 1
        

print('Building Model')

input_ = Input(shape=(max_len_sample, max_len_audio))
#x = Embedding(max_length_audio, EMBEDDING_DIM, input_length=max_length_audio)(input_)
x = Conv1D(128,1,activation='relu')(input_)
x = Conv1D(128,1,activation='relu')(x)
x = MaxPooling1D(1)(x)
#x = Conv1D(128,1,activation='relu')(x)
#x = MaxPooling1D(1)(x)
#x = TimeDistributed(Flatten())(x)
#x = TimeDistributed(Dropout(0.5))(x)
#x = Flatten()(x)
#x = ConvLSTM2D(64, (3,3), activation='relu')(x)
#print(x.shape)
#dense1 = Dense(20, activation='softmax')
encoder1 = Bidirectional(LSTM(LATENT_DIM, return_sequences=True))
x2 = encoder1(x)
#x2 = GlobalMaxPool1D()(x2)
encoder = LSTM(LATENT_DIM, return_state=True)
encoder_output, h, c = encoder(x2)
encoder_states = [h,c]

decoder_inputs_placeholder = Input(shape=(max_len_target,))

#decoder_embedding = Embedding(num_words, EMBEDDING_DIM)   #latent dim
decoder_inputs_x = embedding_layer(decoder_inputs_placeholder)

decoder_lstm = LSTM(LATENT_DIM, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs_x,
    initial_state=encoder_states
)

# final dense layer for predictions
decoder_dense1 = Dense(256, activation='relu')
decoder_outputs = decoder_dense1(decoder_outputs)
decoder_dense2 = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense2(decoder_outputs)

# create the model
model = Model([input_, decoder_inputs_placeholder], decoder_outputs)

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
r = model.fit(
    [padded_fft, decoder_inputs], decoder_target_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0
)

# plot some data
plt.plot(r.history['loss'], label='loss')
#plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
#plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()



############   DECODER   ################
encoder_model = Model(input_, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h , decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = embedding_layer(decoder_inputs_single)

decoder_outputs, h, c = decoder_lstm(
    decoder_inputs_single_x,
    initial_state=decoder_states_inputs
)

decoder_states = [h, c]
decoder_outputs = decoder_dense1(decoder_outputs)
decoder_outputs = decoder_dense2(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

idx2word = {v:k for k,v in word2idx_output.items()}

# print('Saving Model....')
# #serialize model to JSON
# model_json = model.to_json()
# with open('model/model.json', 'w') as json_file:
#     json_file.write(model_json)
# #serialize weights to HDF5
# model.save_weights('model/model.h5')
# print('saved model to disc')
# #seralize model to YAML
# model_yaml = model.to_yaml()
# with open('model/model.yaml', 'w') as yaml_file:
#     yaml_file.write(model_yaml)
  
def nearest_word(input_word):
    input_word_list = [x for x in input_word]
    check = []
    for word in WORDS:
        check_num=0
        for i, x in enumerate(word):
            if word[0]==input_word_list[0]:
                try:
                    if x==input_word_list[i]: check_num+=1
                except: pass
            else: pass
        check.append(check_num)
    return str(WORDS[check.index(max(check))])

    
def decode_sequence(input_seq, path):
    print(input_seq.shape)
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1))
    target_seq[0,0] = word2idx_output['<sos>']
    eos = word2idx_output['<eos>']

    output_word = []
    for _ in range(max_len_target):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value
            )
        idx = np.argmax(output_tokens[0,0,:])
        target_seq=np.zeros((1,1))
        target_seq[0,0] = idx
        #print(idx)
        if eos==idx:
            break
        word = ''
        if idx>0:
            word = idx2word.get(idx)
            #print(word)
            output_word.append(word)

        target_seq[0,0] = idx

        states_value = [h, c]
       
    output_word = ''.join(output_word)
    print(output_word)
    
    if output_word not in WORDS:
        output_word= nearest_word(output_word)
        print(output_word)
        
    return output_word

def predict_word(path):

    #filter_bank_test = get_audio_filter_banks(path_test) 
    #filter_bank_test = filter_bank_test.reshape((1,len(filter_bank_test),40))
    #filter_bank_test_seq = pad_sequences(filter_bank_test,maxlen=max_length_audio,padding='post')
    test_padded_fft = []
    padded_sample_fft = []
    test_fft = get_fft(path, sr=16000, duration=0.20)
    for i, audio in enumerate(test_fft):
        padded_sample_fft.append(librosa.util.pad_center(audio, max_len_audio, axis=0))
    
    zeros = np.zeros((max_len_audio,), dtype='float32')
    range_ = max_len_sample - len(test_fft)
    for _ in range(range_): 
        padded_sample_fft.append(zeros)
    
    test_padded_fft.append(np.array(padded_sample_fft))
    test_padded_fft = np.array(test_padded_fft)
    
    transcribe_text = decode_sequence(test_padded_fft, path)
    return transcribe_text

while True:
    try:
        path = input('Enter the path or n to exit: ')
    except:
        continue
    if path and path.lower().startswith('n'):
        break
    print(predict_word(path))
#ranscribe.append(transcribe_text)
# =============================================================================
# with open('transcribe.txt', 'w') as f:
#     for text in transcribe:
#         f.write(text+'\n')
# =============================================================================
