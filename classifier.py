import keras as ks


def get_model():
    vocab_size = 1000   # we'll define only 1000 unique words in our vocabulary; not a lot of words needed
    num_features = 40   # 40 features on each word vector, same amount of features as spotify's latent audio space
    input_length = 200  # input length of 200 words

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

    # Dropout layer to help in creating connections within the actual network
    model.add(ks.layers.Dropout(rate=dropout_rate))

    # Dense fully connected layer with input: num_features*2 == 80
    model.add(ks.layers.Dense(units=num_features*2))

    model.add(ks.layers.Dense(units=2))



