import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
import csv
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import gensim
from gensim.models import word2vec


from util import DataManager

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('model')
parser.add_argument('action', choices=['train', 'test', 'semi', 'token'])

# training argument
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--nb_epoch', default=10, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
parser.add_argument('--vocab_size', default=20000, type=int)
parser.add_argument('--max_length', default=50, type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM', 'GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=256, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
parser.add_argument('--threshold', default=0.1, type=float)
parser.add_argument('--token')
parser.add_argument('--w2v')
parser.add_argument('--test_path')
parser.add_argument('--train_path')
parser.add_argument('--semi_path')

# output path for your prediction
parser.add_argument(
    '--result_path',
    default='result.csv',
)

# put model in the same directory
parser.add_argument('--load_model', default=None)
parser.add_argument('--save_dir', default='model/')
args = parser.parse_args()

train_path = args.train_path
test_path = args.test_path
semi_path = args.semi_path


# build model
def simpleRNN(args):
    #
    w2v = word2vec.Word2Vec.load(args.w2v)
    token = pk.load(open(args.token, 'rb'))
    vocab = len(token.word_index) + 1
    print(vocab)
    embedding = np.zeros((vocab, 256))
    for word, i in token.word_index.items():
        if word in w2v.wv:
            embedding[i] = w2v.wv[word]
    model = Sequential()
    #model.add(Input(shape=(args.max_length, )))
    model.add(Embedding(vocab, 256, input_length=50, weights=[embedding], trainable=False))
    model.add(Bidirectional(LSTM(args.hidden_size, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(LSTM(args.hidden_size, dropout=0.3)))
    model.add(Dropout(0.5))
    model.add(Dense(
        args.hidden_size // 2,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    #model.add(Activation("sigmoid"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    print('compile model...')

    return model


def main():
    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(get_session(args.gpu_fraction))

    save_path = os.path.join(args.save_dir, args.model)
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir, args.load_model)

#####read data#####
    dm = DataManager()
    print('Loading data...')
    if args.action == 'test':
        dm.add_data('test_data', test_path, False)
    else:
        dm.add_data('train_data', train_path, True)
        #dm.add_data('semi_data', semi_path, False)

    # prepare tokenizer
    print('get Tokenizer...')
    if args.action == 'token':
        dm.tokenize()

    else:
        # read exist tokenizer
        dm.load_tokenizer(args.token)
    '''else:
        # create tokenizer on new data
        dm.tokenize()'''

    dm.save_tokenizer(args.token)

    # convert to sequences
    if args.action != 'token':
        dm.to_sequence(args.max_length)

    # initial model
    if args.action != 'token':
        print('initial model...')
        model = simpleRNN(args)
        print(model.summary())
        if args.load_model is not None:
            if args.action == 'train':
                print('Warning : load a exist model and keep training')
            path = os.path.join(load_path, 'model.h5')
            if os.path.exists(path):
                print('load model from %s' % path)
                model.load_weights(path)
            else:
                raise ValueError("Can't find the file %s" % path)
        elif args.action == 'test':
            print('Warning : testing without loading any model')

    # training
    if args.action == 'train':
        (X, Y), (X_val, Y_val) = dm.split_data('train_data', args.val_ratio)
        earlystopping = EarlyStopping(
            monitor='val_acc', patience=11, verbose=1, mode='max')

        save_path = os.path.join(save_path, 'model.h5')
        checkpoint = ModelCheckpoint(
            filepath=save_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_acc',
            mode='max')
        history = model.fit(
            X,
            Y,
            validation_data=(X_val, Y_val),
            epochs=args.nb_epoch,
            batch_size=args.batch_size,
            callbacks=[checkpoint, earlystopping])

# testing
    elif args.action == 'test':
        X = dm.get_data('test_data')[0]
        predict = model.predict(X)
        result = [['id', 'label']]
        for i in range(len(predict)):
            a = [i]
            if predict[i][0] > 0.5:
                a.append(1)
            else:
                a.append(0) 
            #a.append(predict[i][0])  #test
            #a.append(predict[i])
            result.append(a)
            i += 1
        cout = csv.writer(open(args.result_path, 'w'))
        cout.writerows(result)
        #implement after ensure output format

# semi-supervised training
    elif args.action == 'semi':
        (X, Y), (X_val, Y_val) = dm.split_data('train_data', args.val_ratio)

        [semi_all_X] = dm.get_data('semi_data')
        earlystopping = EarlyStopping(
            monitor='val_acc', patience=11, verbose=1, mode='max')

        save_path = os.path.join(save_path, 'model.h5')
        checkpoint = ModelCheckpoint(
            filepath=save_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_acc',
            mode='max')
        # repeat 10 times
        
        #for i in range(10):
            # label the semi-data
        semi_pred = model.predict(
            semi_all_X, batch_size=1024, verbose=True)
        semi_X, semi_Y = dm.get_semi_data(
            'semi_data', semi_pred, args.threshold, args.loss_function)
        semi_X = np.concatenate((semi_X, X))
        semi_Y = np.concatenate((semi_Y, Y))
        #print('-- iteration %d  semi_data size: %d' % (i + 1, len(semi_X)))
        # train
        history = model.fit(
            semi_X,
            semi_Y,
            validation_data=(X_val, Y_val),
            epochs=20,
            batch_size=args.batch_size,
            callbacks=[checkpoint, earlystopping])

        if os.path.exists(save_path):
            print('load model from %s' % save_path)
            model.load_weights(save_path)
        else:
            raise ValueError("Can't find the file %s" % path)

if __name__ == '__main__':
    main()
