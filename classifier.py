import keras
import scraper as sp
import json
import os
from os import getcwd


# a class to put the email classifier into so that it can run
class EmailClassifierModel(object):


    def __init__(self, vocab_size=400, num_features=40, input_length=2000, dropout_rate=0.3,
                 model=None, index_file=None, data_file=None, model_file=None, load_model=True):
        """

        :param int vocab_size: Maximum length of total vocabulary learned from data
        :param int num_features: Number of Features word vectors will have
        :param int input_length: Length of input for the actual model
        :param float dropout_rate: Floating point number between 1 and 0 for the probability of dropping neural connections in Fully Connected layer
        :param keras.Sequential model: Existing Model if one was created
        :param str index_file: Filepath to the word index JSON file where word serializations will be saved/loaded from
        :param str data_file: Filepath to the training data from where to load training sets from
        :param str model_file: Filepath to where the model should be loaded from or saved to
        :param bool load_model: Flag to indicate whether we should ignore a model file if it was already saved or if we load it
        """

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
                self.model = keras.models.load_model(filepath=self.model_file)
            else:
                self.model = self.create_model() if model is None else model
        # Just create a new compiled model if they don't want us to
        else:
            self.model = self.create_model() if model is None else model

        # set the data file
        self.data_file = "{}/training_data.txt".format(getcwd()) if data_file is None else data_file


        # Create the tokenizer
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

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

                    self.set_word_index_from_data(data, overwrite=True)

        # If the index file doesn't exist, we should do nothing because it should learn the word indexes
        # Through the actual training process since the index file was not specified
        '''
        else:
            data, labels = sp.get_data_from_file(infile=self.data_file, shuffle=False)

            self.tokenizer.fit_on_texts(data)

            with open(self.index_file, 'w') as outfile:
                json.dump(self.tokenizer.word_index, fp=outfile, ensure_ascii=False)
        '''

    ''' Utility Methods '''

    # Create a compiled keras model
    def create_model(self, vocab_size=None, num_features=None, input_length=None, dropout_rate=None):

        :param int vocab_size: Maximum number of words to be learned in embedding layer
        :param int num_features: Dimensionality of embedded word vectors, I.E. the number of features they have
        :param int input_length: Length of input text
        :param float dropout_rate: Floating point number on [0, 1) that indicates the percentage of dropped neuron connections
        :return model: Keras Sequential object
        """
        model = keras.Sequential()
        # input is going to be
        model.add(keras.layers.Embedding(input_dim=vocab_size,  # for the vocabulary size
                                      output_dim=num_features,  # our output is going to be
                                      input_length=input_length))  # (features x input_length)

        # input: (input_length x features) == 200 x 40
        model.add(keras.layers.Conv1D(filters=num_features,  # We use the same amount of filters as features
                                   kernel_size=5,  # kernel_size = 5 for a window of +- 2 words away
                                   padding='same'))  # use 0-padding

        # input: (input_length x features) == 200 x 40
        model.add(keras.layers.MaxPooling1D(pool_size=4))

        # input: (input_length/4 x features) == 50 x 40
        model.add(keras.layers.Conv1D(filters=num_features * 2, kernel_size=5, padding='same', activation='relu'))

        # input: (input_length/4 x features*2) == 50 x 80
        model.add(keras.layers.MaxPooling1D(pool_size=4))
        # outputs: (input_length/8 x features*2) == 25 x 80
        #    |
        #    |
        #    V
        # input

        # Flatten layer
        model.add(keras.layers.Flatten())

        # Dropout layer to help in creating connections within the actual network
        model.add(keras.layers.Dropout(rate=dropout_rate))

        # Dense fully connected layer with input: num_features*2 == 80
        model.add(keras.layers.Dense(units=num_features * 2))

        model.add(keras.layers.Dense(units=1, activation='sigmoid'))

        # compile the model using a binary-crossentropy as the loss function since this is a binary classifier
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        return model


    # This method takes data as an input and it serializes it and write it out to a json index file
    # And returns the data as padded sequences
    def set_word_index_from_data(self, data, serial_file=None, overwrite=True):
        """

        :param array data: List of UTF-8 Strings Given as Training Data
        :param str serial_file: Path to Word Index JSON file
        :param bool overwrite: Flag that indicates whether or not the existing data should be overwritten
        :return: Nothing
        """
        # If we haven't already loaded in a word index for our tokenizer
        # We can create a word index by fitting the tokenizer onto the data provided
        if ("word_index" not in dir(self.tokenizer)) or overwrite:

            # Set the tokenizer's word index with the given data
            self.tokenizer.fit_on_texts(data)

            try:
                # set the serial_file once since we will need to refer to it more than once
                serial_file = self.index_file if serial_file is None else serial_file

                # If overwrite was selected then we open in 'w' otherwise we open in 'x'
                with open(serial_file, 'w' if overwrite else 'x') as outfile:
                    json.dump(self.tokenizer.word_index, outfile, ensure_ascii=False)

            except FileExistsError as fee:
                print("File {} already exists, error: [{}]".format(serial_file, fee))

            except PermissionError as pe:
                print("Permission Error Caught by set_word_index_from_data:\n{}".format(pe))

        else:
            print("we have an existing word index and overwrite is turned off")

    ''' Training routines '''


    def train_model_with_data(self, data=None, labels=None, savefile=None, testing_data_split=0.1,
                              overwrite=True, epoch=100, batch=20, verbosity=2):
        """
        :param list data: Arrays of UTF-8 encoded sentences
        :param list labels: Array of 1s and 0s corresponding to positive and negative data-pieces, respectively
        :param string savefile: File Path to save the trained model
        :param float testing_data_split: Float on domain [0, 1) of the percentage of data that should be alloted to testing
        :param bool overwrite: Boolean flag to specify whether or not we should overwrite existing saved data
        :param int epoch: Integer of epochs to run on the given data
        :param int batch: Integer of how much data should we process at a time
        :param int verbosity: Verbosity Level: 2 - Print each epoch, 1 - Progress bar, 0 - Silent
        :return: Nothing
        """
        # Take the modulus of verbosity by 3 since we don't have an enum object
        # that could easily define modes of verbosity for the method,
        # This way we can easily cap it off at 3 so the user doesn't break it
        verbosity %= 3

        # In the case no actual data was specified
        if data is None or labels is None:
            data, labels = sp.get_data_from_file(infile=self.data_file)

        self.set_word_index_from_data(data)


        # The data provided is turned from strings to arrays of integers which map the words within it to
        # a word index so we can train the model to use text embeddings, this is what is passed
        # to the sequences= argument.
        processed_data = self.tokenizer.texts_to_sequences(data)

        # The sequences are then zero-padded to the end if the actual sequence was
        # shorter than the set input length. If it was longer, anything past the 2000th index is dropped
        # The padding starts at the end of the actual sequence and goes until 2000, which is post-padding
        processed_data = keras.preprocessing.sequence.pad_sequences(sequences=processed_data,
                                                                    maxlen=self.input_length,
                                                                    padding="post")

        # This is the actual training step of the process
        self.model.fit(x=processed_data, y=labels, batch_size=batch, verbose=verbosity,
                       epochs=epoch, validation_split=testing_data_split)

        # Save the model
        self.model.save(filepath = self.model_file if savefile is None else savefile,
                        overwrite=overwrite)


    # to train the model with a different datafile
    # As in the train_model_with_data method, the arguments are virtually identical,
    # However instead of passing in data explicitly, we pass in a datafile path to get the data from
    def train_model_with_file(self, datafile=None, savefile=None, testing_split=0.1,
                              overwrite=True, epoch=100, batch=20, verbosity=2):
        """

        :param str datafile: Path to where the training data needs to be loaded from, overrides self.data_file
        :param str savefile: Path to where the trained model will be saved to, overrides self.model_file
        :param float testing_split: Floating point number between 0 and 1 that indicates what portion of the training data will be used for testing the model
        :param bool overwrite: Flag to indicate whether or not we should overwrite existing files
        :param int epoch: Number of times we should train over every sample in the dataset
        :param int batch: Number of samples we should train with at each step
        :param int verbosity: Level of output during training: 2 - Print each epoch, 1 - Progress Bar, 0 - Total Silence
        :return: Nothing
        """
        # Cap the verbosity level off at 2
        verbosity %= 3

        # These are the labels and the data
        data, labels = sp.get_data_from_file(infile = self.data_file if datafile is None else datafile)

        self.set_word_index_from_data(data)

        # The data provided is turned from strings to arrays of integers which map the words within it to
        # a word index so we can train the model to use text embeddings, this is what is passed
        # to the sequences= argument.
        processed_data = self.tokenizer.texts_to_sequences(data)

        # The sequences are then zero-padded to the end if the actual sequence was
        # shorter than the set input length. If it was longer, anything past the 2000th index is dropped
        # The padding starts at the end of the actual sequence and goes until 2000, which is post-padding
        processed_data = keras.preprocessing.sequence.pad_sequences(sequences=processed_data,
                                                                    maxlen=self.input_length,
                                                                    padding="post")

        # Train the model in the same way we did with the train_model_with_data method
        self.model.fit(x=processed_data, y=labels, batch_size=batch, epochs=epoch,
                       verbose=verbosity, validation_split=testing_split)

        # Save the model
        self.model.save(filepath=self.model_file if savefile is None else savefile, overwrite=overwrite)

if __name__ == "__main__":
    d = EmailClassifierModel()

    d.train_model_with_data()

    # print("word_index" in dir(d.tokenizer))



'''


def get_model():

    dropout_rate = 0.3  # dropout rate of 30%

    model = keras.Sequential()
    # input is going to be
    model.add(keras.layers.Embedding(input_dim=vocab_size,         # for the vocabulary size
                                  output_dim=num_features,      # our output is going to be
                                  input_length=input_length))   # (features x input_length)

    # input: (input_length x features) == 200 x 40
    model.add(keras.layers.Conv1D(filters=num_features,            # We use the same amount of filters as features
                               kernel_size=5,                   # kernel_size = 5 for a window of +- 2 words away
                               padding='same'))                 # use 0-padding

    # input: (input_length x features) == 200 x 40
    model.add(keras.layers.MaxPooling1D(pool_size=4))

    # input: (input_length/4 x features) == 50 x 40
    model.add(keras.layers.Conv1D(filters=num_features*2, kernel_size=5, padding='same', activation='relu'))

    # input: (input_length/4 x features*2) == 50 x 80
    model.add(keras.layers.MaxPooling1D(pool_size=4))
    # outputs: (input_length/8 x features*2) == 25 x 80
    #    |
    #    |
    #    V
    # input

    # Flatten layer
    model.add(keras.layers.Flatten())

    # Dropout layer to help in creating connections within the actual network
    model.add(keras.layers.Dropout(rate=dropout_rate))

    # Dense fully connected layer with input: num_features*2 == 80
    model.add(keras.layers.Dense(units=num_features*2))

    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    # compile the model using a binary-crossentropy as the loss function since this is a binary classifier
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model


# Train the model with the data
def train_model_with_data(data=None, labels=None, model=get_model(), epoch=100, batch=20):

    if data is None:
        data, labels = sp.get_data_from_file(create_if_not_found=False)

    # Initialize the tokenizer object with num_words = vocab_size
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)

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
    X_training = keras.preprocessing.sequence.pad_sequences(X_training, maxlen=input_length, padding="post")

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
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)

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
    model = keras.models.load_model(filepath="{}/models/model.h5".format(getcwd()))

    serialized_text = serialize_text(seq)

    print(serialized_text)

    padded_texts = keras.preprocessing.sequence.pad_sequences(serialized_text, maxlen=input_length, padding="post")

    print(padded_texts)

    prediction = model.predict(padded_texts)

    print(prediction)



'''