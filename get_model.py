from numpy import asarray
import numpy as np
import pandas as pd
import torch
from io import open
import unicodedata, re, os, time, sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sari.SARI import SARIsent


def preprocess(path_txt="newsela_articles_20150302.aligned.sents.txt"):
    """
    Deleting rows without simplification, rows with too long sentences
    creating pandas dataframe
    :param path_txt: path to the dataset
    :return: preprocessed dataframe
    """
    path_txt = os.path.join(sys.path[0], path_txt)
    data = pd.read_csv(path_txt, sep='\t', header=None, names=['doc', 'Vh', 'Vs', 'Complex', 'Simple'])
    data = data.dropna(subset=['Simple'])
    # data.drop_duplicates(subset='Complex', keep='first')
    # print("Mean length of the input sentences is {:.3f}".format(
    #     sum([len(sen) for sen in data['Complex']]) / len(data)))
    # print("Mean length of the output sentences is {:.3f}".format(sum([len(sen) for sen in data['Simple'] \
    #                                                                   if type(sen) == str]) / len(data)))
    max_len = max([len(sen) for sen in data['Complex']])
    while max_len > 270:
        max_len = max([len(sen) for sen in data['Complex']])
        data = data[data['Complex'].map(len) != max_len]

    max_len = max([len(sen) for sen in data['Simple']])
    while max_len > 170:
        max_len = max([len(sen) for sen in data['Simple']])
        data = data[data['Simple'].map(len) != max_len]
    # print("\nTotal count of samples - ", len(data))

    # # show lens in words and in chars of sentences
    # lists_chars, list_words = data_distribution(data)
    # list_words.hist(bins=30)
    # plt.show()
    # lists_chars.hist(bins=30)
    # plt.show()

    return data


def data_distribution(data_txt):
    # empty lists
    complex_lens_words = []
    simple_lens_words = []

    complex_lens = []
    simple_lens = []

    # populate the lists with sentence lengths
    for i in data_txt['Complex']:
        complex_lens_words.append(len(i.split()))
        complex_lens.append(len(i))

    for i in data_txt['Simple']:
        simple_lens_words.append(len(i.split()))
        simple_lens.append(len(i))

    length_words_df = pd.DataFrame({'complex': complex_lens_words, 'simple': simple_lens_words})
    length_df = pd.DataFrame({'complex': complex_lens, 'simple': simple_lens})

    return length_df, length_words_df

def read_emb_dict_w2v(inp_lang, vocab_inp_size):
    # wiki2vec
    from numpy import array
    from numpy import asarray
    from numpy import zeros
    from wikipedia2vec import Wikipedia2Vec

    embedding_dim = 100
    wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

    embedding_matrix = np.zeros((vocab_inp_size, embedding_dim))
    for word, index in inp_lang.word_index.items():
        try:
            embedding_vector = wiki2vec.get_word_vector(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        except:
            continue

    del wiki2vec


def read_emb_dict():
    """
    Reading Glove pre-trained embeddings from file
    :return: dictionary - word: vector
    """
    embed_dictionary = dict()
    path = os.path.join(sys.path[0], 'glove.6B.100d.txt')
    glove_file = open(path, encoding="utf8")
    embed_dim = 100
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embed_dictionary[word] = vector_dimensions
    glove_file.close()
    return embed_dictionary, embed_dim

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang, maxlen=100):

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                             padding='post')
    return tensor, lang_tokenizer


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   embeddings_initializer=tf.keras.initializers.Constant(
                                                       embedding_matrix),
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def create_model(units=512, BATCH_SIZE= 128): # if __name__ == "__main__":

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    @tf.function
    def test_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        return batch_loss

    def training(dataset, EPOCHS = 5):
        path_to_enc = "encoder_more_data"
        path_to_dec = "decoder_more_data"
        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
                if batch % 200 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            # if (epoch + 1) % 2 == 0:
                # checkpoint.save(file_prefix='checkpoint')
            # else:
            # enc_hidden = encoder.initialize_hidden_state()
            #   total_loss = 0

            #   for (batch, (inp, targ)) in enumerate(dataset_val.take(steps_per_epoch)):
            #     batch_loss = train_step(inp, targ, enc_hidden)
            #     total_loss += batch_loss
            #   print('Epoch {} Test Loss {:.4f}'.format(epoch + 1,
            #                                     total_loss / steps_per_epoch))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            encoder.save_weights(path_to_enc)
            decoder.save_weights(path_to_dec)
            print('Saved!')

    def simplify(sentence):
        sentence = preprocess_sentence(sentence)
        inputs = []
        for i in sentence.split(' '):
            try:
                inputs.append(inp_lang.word_index[i])
            except:
                inputs.append(0)
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()

            result += targ_lang.index_word[predicted_id] + ' '

            if targ_lang.index_word[predicted_id] == '<end>':
                # print('Input: %s' % (sentence))
                # print('Predicted translation: {}'.format(result))
                return result

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result

    def evaluate(sentence):
        attention_plot = np.zeros((max_length_targ, max_length_inp))
        sentence = preprocess_sentence(sentence)
        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''

        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += targ_lang.index_word[predicted_id] + ' '

            if targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)
        return result, sentence, attention_plot

    def plot_attention(attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()

    def translate(sentence):
        result, sentence, attention_plot = evaluate(sentence)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        plot_attention(attention_plot, sentence.split(' '), result.split(' '))

    def sari_evaluate():
        data_txt = preprocess()
        data_txt = data_txt[['Complex', 'Simple']]

        input_train, input_val, target_train, target_val = train_test_split(list(data_txt['Complex']),
                                                                            list(data_txt['Simple']),
                                                                            test_size=0.1,
                                                                            shuffle=False,
                                                                            random_state=13)
        sentences = pd.DataFrame(list(zip(input_val, target_val)),
                                 columns=['Complex', 'References'])

        sentences = sentences.groupby(['Complex']).agg(lambda x: tuple(x)).applymap(list).reset_index()
        sents, refs = list(sentences['Complex']), list(sentences['References'])
        saris = []
        n = len(sents)
        for i, pair in enumerate(zip(sents, refs)):
            sent, ref = pair
            predicted_sent = simplify(sent)
            saris.append(SARIsent(sent, predicted_sent, ref))
        print(max(saris))
        print(sum(saris) / n)
        return max(saris), sum(saris) / n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_dictionary, embedding_dim = read_emb_dict()

    data_txt = preprocess()
    num_samples = -1 # 10000

    input_train, input_val, target_train, target_val = train_test_split(list(data_txt['Complex']),
                                                                        list(data_txt['Simple']),
                                                                        test_size=0.1,
                                                                        shuffle=False,
                                                                        random_state=13)

    preprocessed_inp = [preprocess_sentence(sent) for sent in input_train[:num_samples]]

    preprocessed_trg = [preprocess_sentence(sent) for sent in target_train[:num_samples]]

    target_tensor, targ_lang = tokenize(preprocessed_trg)
    input_tensor, inp_lang = tokenize(preprocessed_inp)

    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2,
                                                                                                    random_state=13)

    del preprocessed_trg, preprocessed_inp

    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE

    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    embedding_matrix = np.zeros((vocab_inp_size, embedding_dim))
    for word, index in inp_lang.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    del embeddings_dictionary

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, embedding_matrix)

    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    training(dataset, EPOCHS=1)

    return encoder, decoder, max_length_targ, max_length_inp, inp_lang, targ_lang