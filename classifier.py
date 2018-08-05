import keras as ks


def get_model():
    vocab_size = 1000
    num_features = 40
    input_length = 200

    model = ks.Sequential()
    # input is going to be
    model.add(ks.layers.Embedding(input_dim=vocab_size,         # for the vocabulary size
                                  output_dim=num_features,      # our output is going to be
                                  input_length=input_length))   # (features x input_length)

    # input: (features x input_length) == 40 x 200
    model.add(ks.layers.Conv1D(filters=num_features,            # We use the same amount of filters as features
                               kernel_size=5,                   # kernel_size = 5 for a window of +- 2 words away
                               padding='same'))                 # use 0-padding

    # input: (features x input_length) == 40 x 200
    model.add(ks.layers.MaxPooling1D(pool_size=4))

    # input: (features x input_length/4) == 40 x 50
    model.add(ks.layers.Conv1D(filters=num_features*2, kernel_size=5, padding='same', activation='relu'))

    # input: (features*2 x input_length/4) == 80 x 50
    model.add(ks.layers.MaxPooling1D(pool_size=4))

    # outputs: (features*2 x input_length/8) == 80 x 25
