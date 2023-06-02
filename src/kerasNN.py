from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import os
import sys
import pandas as pd
import numpy as np
import keras as k

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def mee_loss_fn(y_true, y_pred):
    return k.backend.sqrt(k.backend.sum(k.backend.square(y_pred - y_true), axis=-1))


# read training data from csv file
training_df = pd.read_csv('../MLcup_problem/ML-CUP22-TR.csv',
                          names=['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'o1', 'o2'])
# remove all columns with NaN values
training_df = training_df.dropna(axis=0)
training_df.drop_duplicates(inplace=True)

# convert inputs to a numpy array
input_data = np.array(training_df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']])
# convert outputs to a numpy array
output_data = np.array(training_df[['o1', 'o2']])

# split data into training and test sets
train_in, test_in, train_out, test_out = train_test_split(input_data, output_data,
                                                          test_size=0.2, random_state=1, shuffle=True)

# k value for k-fold cross validation
splits = 2

kf = KFold(n_splits=splits, shuffle=True)
kf.get_n_splits(train_in)

best_performance = sys.float_info.max
performance_sum = 0

# loop over all the k possible splits
for train_index, test_index in kf.split(train_in):
    # compute training and validation inputs
    X_train, X_val = train_in[train_index], train_in[test_index]
    # compute training and validation output
    y_train, y_val = train_out[train_index], train_out[test_index]

    # build the model
    inputs = k.Input(shape=(9,))
    x = k.layers.Dense(12, activation="tanh", use_bias=True,
                       kernel_initializer=k.initializers.GlorotUniform(seed=0))(inputs)
    x = k.layers.Dense(12, activation="tanh", use_bias=True,
                       kernel_initializer=k.initializers.GlorotUniform(seed=0))(x)
    x = k.layers.Dense(8, activation="selu", use_bias=True,
                       kernel_initializer=k.initializers.HeUniform(seed=0))(x)
    output = k.layers.Dense(2, activation="linear", use_bias=True)(x)

    model = k.Model(inputs, output)
    opt = k.optimizers.SGD(learning_rate=0.002)
    model.compile(optimizer=opt,
                  loss=mee_loss_fn)

    early_stopping = k.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=500,
        min_delta=0,
        restore_best_weights=True
    )
    callbacks = [early_stopping]

    history = model.fit(X_train, y_train,
                        epochs=700,
                        batch_size=4,
                        shuffle=False,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks)

    # keep track of the best model over all the splits
    performance = min(history.history['val_loss'])
    performance_sum += performance
    if performance < best_performance:
        best_performance = performance
        best_model = model
        best_history = history

figure(figsize=(10, 6))
plt.plot(best_history.history['loss'], color='red', label='Training loss curve')
plt.plot(best_history.history['val_loss'], color='blue', linestyle='dashed', label='Validation loss curve')
plt.xlabel('Epochs', fontsize='20')
plt.xticks(fontsize=15)
plt.ylabel('Loss (MEE)', fontsize='20')
plt.yticks(fontsize=15)
plt.legend(fontsize='20')
plt.show()
# compute the performance of the best model on the validation and test sets
best_model.summary()
performance = performance_sum / splits
print('performance on the validation test: ', performance)
result = best_model.evaluate(test_in, test_out, batch_size=4)
print('performance on the test set: ', result)
