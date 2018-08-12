import keras as ks
import scraper as sp
import json
from os import getcwd

vocab_size = 4000  # we'll define only 1000 unique words in our vocabulary; not a lot of words needed
num_features = 40  # 40 features on each word vector, same amount of features as spotify's latent audio space
input_length = 2000  # input length of 200 words


# a class to put the email classifier into so that it can run
class EmailClassifierModel(object):

    def __init__(self, vocab_size=400, num_features=40, input_length=2000, dropout_rate=0.3, datafile=None):
        self.vocab_size = vocab_size
        self.num_features = num_features
        self.input_length = input_length
        self.dropout_rate = dropout_rate

        if datafile is None:
            if os.exists


def get_model():

    dropout_rate = 0.3  # dropout rate of 30%

    model = ks.Sequential()
    # input is going to be
    model.add(ks.layers.Embedding(input_dim=vocab_size,         # for the vocabulary size
                                  output_dim=num_features,      # our output is going to be
                                  input_length=input_length))   # (features x input_length)

    # input: (input_length x features) == 200 x 40
    model.add(ks.layers.Conv1D(filters=num_features,            # We use the same amount of filters as features
                               kernel_size=5,                   # kernel_size = 5 for a window of +- 2 words away
                               padding='same'))                 # use 0-padding

    # input: (input_length x features) == 200 x 40
    model.add(ks.layers.MaxPooling1D(pool_size=4))

    # input: (input_length/4 x features) == 50 x 40
    model.add(ks.layers.Conv1D(filters=num_features*2, kernel_size=5, padding='same', activation='relu'))

    # input: (input_length/4 x features*2) == 50 x 80
    model.add(ks.layers.MaxPooling1D(pool_size=4))
    # outputs: (input_length/8 x features*2) == 25 x 80
    #    |
    #    |
    #    V
    # input

    # Flatten layer
    model.add(ks.layers.Flatten())

    # Dropout layer to help in creating connections within the actual network
    model.add(ks.layers.Dropout(rate=dropout_rate))

    # Dense fully connected layer with input: num_features*2 == 80
    model.add(ks.layers.Dense(units=num_features*2))

    model.add(ks.layers.Dense(units=1, activation='sigmoid'))

    # compile the model using a binary-crossentropy as the loss function since this is a binary classifier
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model


# Train the model with the data
def train_model_with_data(data=None, labels=None, model=get_model(), epoch=100, batch=20):

    if data is None:
        data, labels = sp.get_data_from_file(create_if_not_found=False)

    # Initialize the tokenizer object with num_words = vocab_size
    tokenizer = ks.preprocessing.text.Tokenizer(num_words=vocab_size)

    # Tokenize the text that we actually have so the words map to integers
    tokenizer.fit_on_texts(data)

    # We should save the data that we created into a dictionary for later re-use if needed
    try:
        with open('word_indices.json', 'x') as outfile:
            # dump the word index to a json file and make sure nothing is converted to ascii
            # due to issues that arise with trying to tokenize unicode characters
            json.dump(tokenizer.word_index, fp=outfile, ensure_ascii=False)
    except FileExistsError as FEE:
        print(FEE)

    X_training = tokenizer.texts_to_sequences(data)

    # Input length will be 2000 because the average character length is ~1100 for emails with
    # A standard deviation of std = +/- 900 char with as a left-skewed distribution
    X_training = ks.preprocessing.sequence.pad_sequences(X_training, maxlen=input_length, padding="post")

    # print(X_training)

    model.fit(x=X_training, y=labels, epochs=epoch, batch_size=batch)

    loss, accuracy = model.evaluate(x=X_training, y=labels, batch_size=batch)

    print("Accuracy: {:0.3%}%\nLoss: {:3f}".format(accuracy, loss))

    # save the model
    model.save(filepath="{}/models/first_model.h5".format(getcwd()))


def serialize_text(text, filepath="word_indices.json"):
    with open(file=filepath, mode='r') as infile:
        vocab = json.load(infile)

    for key, value in vocab.items():
        print("{}: {}".format(key, value))



'''
# evaluate the model and return a prediction
def predict(text, filepath="{}/models/first_model.h5".format(getcwd()), model=None):
'''


if __name__ == '__main__':
    '''data, label = sp.get_data_from_file(create_if_not_found=False)
    for d in data:
        print(d)
    train_model_with_data(data, label)
    '''
    serialize_text("")
    # train_model_with_data(data=data, labels=label)




