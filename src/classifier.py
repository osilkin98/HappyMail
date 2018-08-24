import keras
from src import scraper
import json
import os
from os import getcwd
from tensorflow.python.framework.errors_impl import InternalError as TFInternalError


# a class to put the email classifier into so that it can run
class EmailClassifierModel(object):

    def __init__(self, vocab_size=400, num_features=40, input_length=2000, dropout_rate=0.3, epochs=100,
                 model_dir=None, logging_dir=None, model=None, index_file=None, data_file=None, model_file=None,
                 auto_train=True, load_model=True):
        """
        :param int vocab_size: Maximum length of total vocabulary learned from data
        :param int num_features: Number of Features word vectors will have
        :param int input_length: Length of input for the actual model
        :param float dropout_rate: Floating point number between 1 and 0 for the probability of dropping neural connections in Fully Connected layer
        :param int epoch: Number of times to run the model through the entire training set
        :param str model_dir: Directory to where the model file should be saved, uses ./models if None is specified
        :param str logging_dir: Directory to where the TensorFlow logging files will be saved, uses ./models/logs if None is specified
        :param keras.Sequential model: Existing Model if one was created
        :param str index_file: Filepath to the word index JSON file where word serializations will be saved/loaded from
        :param str data_file: Filepath to the training data from where to load training sets from
        :param str model_file: Filepath to where the model should be loaded from or saved to
        :param bool auto_train: Flag to indicate whether or not the model should be retrained if we couldn't load it
        :param bool load_model: Flag to indicate whether we should ignore a model file if it was already saved or if we load it
        """

        # To set the model's hyper-parameters
        self.vocab_size = vocab_size
        self.num_features = num_features
        self.input_length = input_length
        self.dropout_rate = dropout_rate
        self.epochs = epochs

        # Sets the directory for where the file should be saved
        self.model_dir = model_dir if model_dir is not None else "{}/models".format(getcwd())

        # Sets the directory for tensorflow logging
        self.logging_dir = logging_dir if logging_dir is not None else "{}/logs".format(self.model_dir)

            # Set the actual model file, if it's an absolute file then it overrides self.model_dir
        self.model_file = "{}/model.h5".format(self.model_dir) if model_file is None \
                            else model_file if model_file[-1] == '/' else "{}/{}".format(getcwd(), model_file)



        # If the model path doesn't exist and we weren't passed an absolute path then we create the directories
        if self.model_file[-1] != '/' and not os.path.exists(self.model_dir):
            print("making directories: {}".format(self.model_dir))
            os.makedirs(self.model_dir)

        self.trained = False

        # Load the model if the user wants to
        if load_model:
            # If the file already exists, we'll load the model, otherwise we'll just recreate it
            if os.path.exists(self.model_file):
                try:
                    self.model = keras.models.load_model(filepath=self.model_file)
                    # Assuming we load the model correctly, it will be trained
                    self.trained = True
                    print("Loaded model")

                except TFInternalError as tfe:
                    print("Encountered {}, creating new model from scratch.".format(tfe))
                    self.model = self.create_model(self.vocab_size, self.num_features,
                                                   self.input_length, self.dropout_rate)
            else:
                print("Creating a model here")
                self.model = self.create_model(vocab_size=self.vocab_size,
                                               num_features=self.num_features,
                                               input_length=self.input_length,
                                               dropout_rate=self.dropout_rate) if model is None else model
        # Just create a new compiled model if they don't want us to
        else:
            self.model = self.create_model(vocab_size=self.vocab_size,
                                           num_features=self.num_features,
                                           input_length=self.input_length,
                                           dropout_rate=self.dropout_rate) if model is None else model

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

                    data, labels = scraper.get_data_from_file(infile=self.data_file, shuffle=False)

                    self.set_word_index_from_data(data, overwrite=True)

        if not self.trained and auto_train:
            self.train_model_with_data(epoch=epochs)

        # If the index file doesn't exist, we should do nothing because it should learn the word indexes
        # Through the actual training process since the index file was not specified
        '''
        else:
            data, labels = scraper.get_data_from_file(infile=self.data_file, shuffle=False)

            self.tokenizer.fit_on_texts(data)

            with open(self.index_file, 'w') as outfile:
                json.dump(self.tokenizer.word_index, fp=outfile, ensure_ascii=False)
        '''

    ''' Utility Methods '''

    @staticmethod
    def create_model(vocab_size=4000, num_features=40, input_length=2000, dropout_rate=0.3):
        """

        :param vocab_size: Maximum number of words to be learned in embedding layer
        :type vocab_size: int
        :param num_features: Dimensionality of embedded word vectors, I.E. the number of features they have
        :type num_features: int
        :param input_length: Length of input text
        :type input_length: int
        :param dropout_rate: Floating point number on [0, 1) that indicates the percentage of dropped neuron connections
        :type dropout_rate: float
        :return model: Keras Sequential object
        """
        model = keras.Sequential()
        # input is going to be
        # input is going to be
        model.add(keras.layers.Embedding(input_dim=vocab_size,  # for the vocabulary size
                                         output_dim=num_features,  # our output is going to be
                                         input_length=input_length,
                                         mask_zero=True))  # (features x input_length)

        model.add(keras.layers.LSTM(num_features, dropout=dropout_rate))

        '''
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
        '''

        # Dropout layer to help in creating connections within the actual network
        # model.add(keras.layers.Dropout(rate=dropout_rate))

        # Dense fully connected layer with input: num_features*2 == 80
        # model.add(keras.layers.Dense(units=num_features * 2))

        model.add(keras.layers.Dense(units=1, activation='sigmoid'))

        # compile the model using a binary-crossentropy as the loss function since this is a binary classifier
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

        return model

    # This method takes data as an input and it serializes it and write it out to a json index file
    # And returns the data as padded sequences
    def set_word_index_from_data(self, data, serial_file=None, overwrite=True):
        """

        :param list data: List of UTF-8 Strings Given as Training Data
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

    def train_model_with_data(self, data=None, labels=None, savefile=None, load=True, testing_data_split=0.1,
                              log_dir=None, overwrite=True, epoch=50, batch=64, verbosity=2):
        """
        :param list data: Arrays of UTF-8 encoded sentences
        :param list labels: Array of 1s and 0s corresponding to positive and negative data-pieces, respectively
        :param string savefile: File Path to save the trained model, overrides self.model_file
        :param float testing_data_split: Float on domain [0, 1) of the percentage of data that should be alloted to testing
        :param str log_dir: Directory for where to log the TensorFlow graph, overrides the default object's directory
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
            data, labels = scraper.get_data_from_file(infile=self.data_file)

        self.set_word_index_from_data(data)


        # The data provided is turned from strings to arrays of integers which map the words within it to
        # a word index so we can train the model to use text embeddings, this is what is passed
        # to the sequences= argument.
        processed_data = self.tokenizer.texts_to_sequences(data)

        # The sequences are then zero-padded to the end if the actual sequence was
        # shorter than the set input length. If it was longer, anything past the 2000th index is dropped
        # The padding starts at the end of the actual sequence and goes until 2000, which is post-padding
        ''' Since we will try and use CuDDNLSTM, we will try and do away with the padding & zero-masking'''

        processed_data = keras.preprocessing.sequence.pad_sequences(sequences=processed_data,
                                                                    maxlen=self.input_length,
                                                                    padding="post")

        log_dir = self.logging_dir if log_dir is None else \
            "{}/{}".format(getcwd(), log_dir) if log_dir[0] != '/' else log_dir

        load_progress = ""
        # Try and load
        if load and os.path.exists(savefile if savefile is not None else self.model_file):
            load_progress += "Loading... "
            try:
                self.model = keras.models.load_model(filepath=savefile if savefile is not None else self.model_file)
                load_progress += "done\n"

            except ValueError as ve:
                load_progress += "failed, invalid savefile\n"
                self.model = self.create_model(self.vocab_size,
                                               self.num_features,
                                               self.input_length,
                                               self.dropout_rate)
            finally:
                print(load_progress)
        self.model.summary()

        print("Fitting the model...")
        print(processed_data)
        # This is the actual training step of the process
        self.model.fit(x=processed_data, y=labels, batch_size=batch, verbose=verbosity,
                       epochs=epoch, validation_split=testing_data_split,
                       callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir)])

        self.model.evaluate(x=processed_data, y=labels)
        print("Done")


        # Save the model
        if not os.path.exists(self.model_dir) and savefile is None:
            os.mkdir(self.model_dir)


        self.model.save(filepath = self.model_file if savefile is None else savefile,
                        overwrite=overwrite)


    # to train the model with a different datafile
    # As in the train_model_with_data method, the arguments are virtually identical,
    # However instead of passing in data explicitly, we pass in a datafile path to get the data from
    def train_model_with_file(self, datafile= None, savefile = None, testing_split = 0.1,
                              overwrite = True, epoch = 100, batch = 20, verbosity = 2):

        """ Train the model with the given file

        :param str datafile: Path to where the training data needs to be loaded from, overrides self.data_file
        :param str savefile: Path to where the trained model will be saved to, overrides self.model_file
        :param float testing_split: Floating point number between 0 and 1 that indicates what portion of the training data will be used for testing the model
        :param bool overwrite: Flag to indicate whether or not we should overwrite existing files
        :param int epoch: Number of times we should train over every sample in the dataset
        :param int batch: Number of samples we should train with at each step
        :param int verbosity: Level of output during training: 2 - Print each epoch, 1 - Progress Bar, 0 - Total Silence
        :return: None
        """
        # Cap the verbosity level off at 2
        verbosity %= 3

        # These are the labels and the data
        data, labels = scraper.get_data_from_file(infile = self.data_file if datafile is None else datafile)

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

    def predict(self, texts: tuple):
        """

        :param tuple texts: A list/tuple of texts to compute predictions on
        :return: Returns a listed of computed values between 0 and 1, 1 being positive and 0 being negative. These Texts correspond to the index of the text string
        """

        # if we don't have any word serialization
        if "word_index" not in dir(self.tokenizer):
            try:
                with open(self.index_file, mode='r') as infile:
                    self.tokenizer.word_index = json.load(infile)
            except IOError as e:
                print("IOError encountered while opening {}: {}".format(self.index_file, e))
                return -1

        # otherwise we should be fine to go

        # We enclose the text given as a single element within an array
        to_process = self.tokenizer.texts_to_sequences(texts)

        # We then pad the sequence to the end with 0s if text < 2000 and cut it off at 2000 if text > 2000
        to_process = keras.preprocessing.sequence.pad_sequences(to_process,
                                                                maxlen=self.input_length,
                                                                padding='post')

        return self.model.predict(to_process)


def test_class(ModelObject):

    while True:
        to_input = input("Enter an email snippet [max 200 chars]: ")

        pred = ModelObject.predict(to_input)

        print(pred)


def function(param1, param2):
    """

    :param int param1: First parameter
    :param tuple param2: Second Paramter
    :return: Returns some shit here
    """


if __name__ == "__main__":
    d = EmailClassifierModel(input_length=2000, vocab_size=5000, model_file="models/trained_net.h5", epochs=400)

    print(d.__dict__)

    test_class(d)



















