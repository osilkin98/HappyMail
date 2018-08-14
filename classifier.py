import keras as ks
import scraper as sp
import json
import os
from os import getcwd

vocab_size = 4000  # we'll define only 1000 unique words in our vocabulary; not a lot of words needed
num_features = 40  # 40 features on each word vector, same amount of features as spotify's latent audio space
input_length = 2000  # input length of 200 words
'''
'''
# a class to put the email classifier into so that it can run
class EmailClassifierModel(object):

    def __init__(self, vocab_size=400, num_features=40, input_length=2000, dropout_rate=0.3,
                 model=None, index_file=None, data_file=None, model_file=None, load_model=True):

        # To set the model's hyper-parameters
        self.vocab_size = vocab_size
        self.num_features = num_features
        self.input_length = input_length
        self.dropout_rate = dropout_rate

        # Set the actual model file
        self.model_file = "{}/models/model.h5" if model_file is None else model_file

        # Load the model if the user wants to
        if load_model:
            # If the file already exists, we'll load the model, otherwise we'll just recreate it
            if os.path.exists(self.model_file):
                self.model = ks.models.load_model(filepath=self.model_file)
            else:
                self.model = self.create_model() if model is None else model
        # Just create a new compiled model if they don't want us to
        else:
            self.model = self.create_model() if model is None else model

        # set the data file
        self.data_file = "{}/training_data.txt".format(getcwd()) if data_file is None else data_file


        # Create the tokenizer
        self.tokenizer = ks.preprocessing.text.Tokenizer(num_words=self.vocab_size)

        self.index_file = "{}/word_indices.json".format(getcwd()) if index_file is None else index_file

        # Try to set the indices if we already have the file
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as infile:
                    self.tokenizer.word_index = json.load(infile)

            # if something went wrong we should try and just load the data
            except json.JSONDecodeError as de:

                # If we have an existing data file, we should try to overwrite the word index file
                if os.path.exists(self.data_file):
                    print(de)

                    data, labels = sp.get_data_from_file(infile=self.data_file, shuffle=False)

                    self.tokenizer.fit_on_texts(data)

                    print("Since {} couldn't be read from correctly, we will try and overwrite it".format(
                        self.index_file))

                    try:
                        with open(self.index_file, 'w') as outfile:
                            json.dump(self.tokenizer.word_index, fp=outfile, ensure_ascii=False)

                    # in case soemthing goes wrong again
                    except PermissionError as pe:
                        print(pe)

        # If the index file doesn't exist, we should do nothing because it should learn the word indexes
        # Through the actual training process since the index file was not specified
        '''
        else:
            data, labels = sp.get_data_from_file(infile=self.data_file, shuffle=False)

            self.tokenizer.fit_on_texts(data)

            with open(self.index_file, 'w') as outfile:
                json.dump(self.tokenizer.word_index, fp=outfile, ensure_ascii=False)
        '''



    # Create a compiled keras model
    def create_model(self, vocab_size=None, num_features=None, input_length=None, dropout_rate=None):

        if vocab_size is None:
            vocab_size = self.vocab_size

        if num_features is None:
            num_features = self.num_features

        if input_length is None:
            input_length = self.input_length

        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        model = ks.Sequential()
        # input is going to be
        model.add(ks.layers.Embedding(input_dim=vocab_size,  # for the vocabulary size
                                      output_dim=num_features,  # our output is going to be
                                      input_length=input_length))  # (features x input_length)

        # input: (input_length x features) == 200 x 40
        model.add(ks.layers.Conv1D(filters=num_features,  # We use the same amount of filters as features
                                   kernel_size=5,  # kernel_size = 5 for a window of +- 2 words away
                                   padding='same'))  # use 0-padding

        # input: (input_length x features) == 200 x 40
        model.add(ks.layers.MaxPooling1D(pool_size=4))

        # input: (input_length/4 x features) == 50 x 40
        model.add(ks.layers.Conv1D(filters=num_features * 2, kernel_size=5, padding='same', activation='relu'))

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
        model.add(ks.layers.Dense(units=num_features * 2))

        model.add(ks.layers.Dense(units=1, activation='sigmoid'))

        # compile the model using a binary-crossentropy as the loss function since this is a binary classifier
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        return model


    # train the model with specified data, or the default datafile if none is provided
    def train_model_with_data(self, data=None, labels=None, savefile=None, overwrite=True, epoch=100, batch=20):

        # In the case no actual data was specified
        if data is None or labels is None:
            data, labels = sp.get_data_from_file(infile=self.data_file)

        # If we haven't already loaded in a word index for our tokenizer
        # We can create a word index by fitting the tokenizer onto the data provided
        if "word_index" not in dir(d.tokenizer):
            self.tokenizer.fit_on_texts(data)

            # We should try to write the tokenizer word indices to json index file
            try:
                # we can open it in the mode corresponding to whether or not
                # The user specified if we should overwrite existing data
                with open(self.index_file, mode='{}'.format('w' if overwrite else 'x')) as outfile:
                    json.dump(self.tokenizer.word_index, outfile, ensure_ascii=False)

            except FileExistsError as fee:
                print("File {} already exists, error: [{}]".format(self.index_file, fee))


        processed_data = self.tokenizer.texts_to_sequences(data)
        print(processed_data)



    # to train the model with a different datafile
    def train_model_with_file(self, datafile=None, overwrite=True, epoch=100, batch=20):




'''


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
    model.save(filepath="{}/models/model.h5".format(getcwd()))


# Temporary function will not stay here for long
def serialize_text(text, filepath="word_indices.json"):
    with open(file=filepath, mode='r') as infile:
        vocab = json.load(infile)

    # Create a tokenizer object
    tokenizer = ks.preprocessing.text.Tokenizer(num_words=vocab_size)

    tokenizer.word_index = vocab

    return tokenizer.texts_to_sequences([text])



'''

'''
# evaluate the model and return a prediction
def predict(text, filepath="{}/models/model.h5".format(getcwd()), model=None):
'''
'''
seq = "Dear , Thank you for your application and interest in joining our team; we truly appreciate the time you took to apply. Upon review we found your skills and qualifications to be notable however, at this time, we have identified other candidates whose experience more closely fit the unique specifications of this current opening.While we can no longer consider you for this specific role, we are constantly seeking to strengthen our team with people who can empower the growing Virgin Pulse team to be better and stronger and we hope you will consider applying for other positions which better match your qualifications in the future. If you have applied for other positions, we will continue to review your qualifications and will keep you informed about your status for those opportunities.  Again, we appreciate the time you have taken to apply and wish you the best.Be Well,The Virgin Pulse Recruiting TeamYou can change your email preferences at:https://app.jobvite.com/l?ksJuXChwy"

if __name__ == '__main__':
    '''
    '''data, label = sp.get_data_from_file(create_if_not_found=False)
    for d in data:
        print(d)
    train_model_with_data(data, label)
    ''''''

    print("Loading model")
    model = ks.models.load_model(filepath="{}/models/model.h5".format(getcwd()))

    serialized_text = serialize_text(seq)

    print(serialized_text)

    padded_texts = ks.preprocessing.sequence.pad_sequences(serialized_text, maxlen=input_length, padding="post")

    print(padded_texts)

    prediction = model.predict(padded_texts)

    print(prediction)



'''