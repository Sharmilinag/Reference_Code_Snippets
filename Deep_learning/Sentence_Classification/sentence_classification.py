import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

plt.style.use('ggplot')

filepath_dict = {'yelp': 'data/yelp_labelled.txt',
                 'amazon': 'data/amazon_cells_labelled.txt',
                 'imdb': 'data/imdb_labelled.txt'}
epochs = 20
embedding_dim = 50
maxlen = 100
sentence_tokenizer_max_words = 5000
vocab_size = None  # updated during pre_processing step

model_train_graph = 'binary_classification_acc_loss.png'

# ## set HYPER_PARAMETER_TUNING = 1 for hyper parameter tuning, else 0 ( no hyper parameter tuning )
HYPER_PARAMETER_TUNING = 1


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    # ## plotting the actual graph
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(model_train_graph)


def read_data():
    df = pd.read_csv(filepath_dict['yelp'], names=['sentence', 'label'], sep='\t')
    sentences = df['sentence'].values
    y = df['label'].values
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)
    return sentences_train, sentences_test, y_train, y_test


def build_model(num_filters, kernel_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def pre_process_data(sentences_train, sentences_test):
    global vocab_size
    # Tokenize words
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)
    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1
    # Pad sequences with zeros
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    return X_train, X_test


def run_tuned_model(X_train, X_test, y_train, y_test):
    num_filters = 128
    kernel_size = 5
    embedding_dim = 50
    maxlen = 100
    model = build_model(num_filters, kernel_size, embedding_dim, maxlen)
    history = model.fit(X_train, y_train,
                        epochs=10,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print('\n=====================================================================================\n')
    print(history.history.keys())
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    print('\n=====================================================================================\n')
    plot_history(history)


def run_hyper_parameter_tuning(X_train, X_test, y_train, y_test):
    # Parameter grid for grid search
    param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      embedding_dim=[embedding_dim],
                      maxlen=[maxlen])
    model = KerasClassifier(build_fn=build_model,
                            epochs=epochs, batch_size=10,
                            verbose=False)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)
    grid_result = grid.fit(X_train, y_train)
    # Evaluate testing set
    test_accuracy = grid.score(X_test, y_test)
    print('\n=====================================================================================\n')
    s = 'Best Training Accuracy : {:.4f}\nBest Params : {}\nTest Accuracy : {:.4f}\n'
    output_string = s.format(
        grid_result.best_score_,
        grid_result.best_params_,
        test_accuracy)
    print(output_string)
    print('\n=====================================================================================\n')


def main():
    sentences_train, sentences_test, y_train, y_test = read_data()
    X_train, X_test = pre_process_data(sentences_train, sentences_test)
    if HYPER_PARAMETER_TUNING == 0:
        run_tuned_model(X_train, X_test, y_train, y_test)
    else:
        run_hyper_parameter_tuning(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
