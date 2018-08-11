import keras as ks
import scraper as sp
import json
from os import getcwd

vocab_size = 4000  # we'll define only 1000 unique words in our vocabulary; not a lot of words needed
num_features = 40  # 40 features on each word vector, same amount of features as spotify's latent audio space
input_length = 2000  # input length of 200 words

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
def train_model_with_data(data=None, labels=None, model=get_model(), epoch=100):

    if data is None:
        data, labels = sp.get_data_from_file(create_if_not_found=False)









if __name__ == '__main__':
    my_model = get_model()
    print(my_model.summary())




