import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.utils import np_utils
import pickle
import matplotlib.pyplot as plt
import warnings

plt.style.use('ggplot')
warnings.filterwarnings('ignore')

data_path = 'data/multiclass_consumer_complaints.csv'
WORD_EMBEDDING_PATH = 'glove.6B/glove.6B.50d.txt'

epochs = 20
embedding_dim = 50
maxlen = 100
sentence_tokenizer_max_words = 5000
vocab_size = None  # updated during pre_processing step
output_class_cnt = None  # updated during read_data step

model_train_graph = 'multiclass_classification_acc_loss.png'

# ## to train model make TRAIN_MODEL = 1
TRAIN_MODEL = 1


def plot_history(history):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
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


def build_model(num_filters, kernel_size, embedding_dim, maxlen, embedding_matrix=None):
    model = Sequential()
    if embedding_matrix is None:
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    else:
        model.add(layers.Embedding(vocab_size, embedding_dim,
                                   weights=[embedding_matrix],
                                   input_length=maxlen,
                                   trainable=True))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(11, activation='relu'))
    model.add(layers.Dense(output_class_cnt, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def run_tuned_model(X_train, X_test, y_train, y_test, embedding_matrix=None):
    num_filters = 128
    kernel_size = 5
    embedding_dim = 50
    maxlen = 100
    if embedding_matrix is None:
        model = build_model(num_filters, kernel_size, embedding_dim, maxlen)
    else:
        model = build_model(num_filters, kernel_size, embedding_dim, maxlen, embedding_matrix)
    # train model
    print('-------- Model Training Started -----------')
    history = model.fit(X_train, y_train,
                        epochs=20,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=10)
    print('-------- Model Training Completed -----------')
    # save model and architecture to single file
    model.save("model.h5")
    # evaluate trained model
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print('\n=====================================================================================\n')
    print(history.history.keys())
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    print('\n=====================================================================================\n')
    plot_history(history)
    return model


def read_data():
    global output_class_cnt
    df = pd.read_csv(data_path, usecols=['consumer_complaint_narrative', 'product'], low_memory=False)
    df = df.head(10000)
    output_class_cnt = len(df['product'].unique())
    print('Total Classes Recognized  =  ' + str(output_class_cnt))
    sentences = df['consumer_complaint_narrative'].values
    y = df['product'].values
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)
    return sentences_train, sentences_test, y_train, y_test


def pre_process_data(sentences_train, sentences_test, y_train, y_test, label_encoder):
    global vocab_size
    # Tokenize words
    tokenizer = Tokenizer(num_words=sentence_tokenizer_max_words)
    tokenizer.fit_on_texts(sentences_train)
    # saving tokenizer for predictions
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)
    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1
    # Pad sequences with zeros
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    # ## encode labels
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)
    # ## convert integers to dummy variables (i.e. one hot encoded)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, X_test, y_train, y_test, label_encoder, tokenizer


def load_prediction_artifacts():
    # load model
    model = load_model('model.h5')
    # load tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


def make_probability_preds(model, X_pred):
    # make a prediction
    ynew = model.predict_proba(X_pred)
    # show the inputs and predicted outputs
    for i in range(len(X_pred)):
        print("X=%s, Predicted=%s" % (X_pred[i], ynew[i]))


#   ##  Ref: https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
def create_embedding_matrix(filepath, tokenizer, embedding_dim):
    global vocab_size
    embeddings_index = dict()
    f = open(filepath, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, index in tokenizer.word_index.items():
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector


def main():
    le = LabelEncoder()
    if TRAIN_MODEL == 1:
        sentences_train, sentences_test, y_train, y_test = read_data()
        X_train, X_test, y_train, y_test, label_encoder, tokenizer = pre_process_data(sentences_train,
                                                                                      sentences_test,
                                                                                      y_train,
                                                                                      y_test,
                                                                                      le)
        embedding_matrix = create_embedding_matrix(WORD_EMBEDDING_PATH, tokenizer, embedding_dim)
        # ## Train without GLoVE embedding
        # model = run_tuned_model(X_train, X_test, y_train, y_test, None)
        # ## Train with GLoVE embedding
        model = run_tuned_model(X_train, X_test, y_train, y_test, embedding_matrix)
    else:
        # ## read in test_data (data_file for prediction should ideally be different than training data)
        model, tokenizer = load_prediction_artifacts()
        data_test = pd.read_csv(data_path, usecols=['consumer_complaint_narrative'], low_memory=False)[0:10]
        X_pred = tokenizer.texts_to_sequences(data_test['consumer_complaint_narrative'].values)
        X_pred = pad_sequences(X_pred, padding='post', maxlen=maxlen)
        make_probability_preds(model, X_pred)


if __name__ == "__main__":
    main()
