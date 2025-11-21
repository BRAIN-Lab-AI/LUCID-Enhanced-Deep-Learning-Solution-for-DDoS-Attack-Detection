import os
import sys
import time
import glob
import csv
import pprint
import argparse
import random as rn
import numpy as np
import pyshark
import tensorflow as tf
from tqdm import tqdm, trange
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Activation, Flatten, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from scikeras.wrappers import KerasClassifier
from util_functions import load_dataset, count_packets_in_dataset, static_min_max
from lucid_dataset_parser import parse_labels, process_live_traffic, dataset_to_list_of_fragments, normalize_and_padding

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} Physical GPUs found and configured.")
    except RuntimeError as e:
        print(e)


OUTPUT_FOLDER = "./output/"
VAL_HEADER = ['Model', 'Samples', 'Accuracy', 'F1Score', 'Hyper-parameters','Validation Set']
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR','TNR', 'FNR', 'Source']
PATIENCE = 10
DEFAULT_EPOCHS = 1000
HYPERPARAMETERS = {
    "learning_rate": [0.01, 0.001],
    "batch_size": [1024, 2048],
    "kernels": [32, 64],
    "regularization": [None, 'l1'],
    "dropout": [None, 0.2],
    "weight_decay": [0.01, 0.001, 0.0],
    "dense_units": [16, 32],     
    "dense_dropout": [0.25, 0.5]
}


def Conv2DModel(model_name, input_shape, kernel_col, kernels=64, kernel_rows=3, learning_rate=0.01, regularization=None, dropout=None, weight_decay=0.0, dense_units=16, dense_dropout=0.25):
    model = Sequential(name=model_name)
    model.add(Input(shape=input_shape))
    
    reg = regularizers.L1(l1=0.01) if regularization == 'l1' else None

    model.add(Conv2D(kernels, (kernel_rows, kernel_col), kernel_regularizer=reg, name='conv0'))
    model.add(BatchNormalization())
    if dropout is not None and isinstance(dropout, float):
        model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dense_dropout))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid', name='fc1'))
    
    compileModel(model, learning_rate, weight_decay)
    return model

def compileModel(model, lr, wd):
    optimizer = AdamW(learning_rate=lr, weight_decay=wd)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer):
    ddos_rate = sum(Y_pred) / Y_pred.shape[0] if Y_pred.shape[0] > 0 else 0

    row = {
        'Model': model_name, 'Time': f'{prediction_time:04.3f}', 'Packets': packets,
        'Samples': Y_pred.shape[0], 'DDOS%': f'{ddos_rate:04.3f}'
    }

    if Y_true is not None:
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()
        
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        row.update({
            'Accuracy': f'{accuracy:05.4f}', 'F1Score': f'{f1:05.4f}',
            'TPR': f'{tpr:05.4f}', 'FPR': f'{fpr:05.4f}', 
            'TNR': f'{tnr:05.4f}', 'FNR': f'{fnr:05.4f}'
        })
    else:
        row.update({
            'Accuracy': "N/A", 'F1Score': "N/A", 'TPR': "N/A", 'FPR': "N/A",
            'TNR': "N/A", 'FNR': "N/A"
        })
    
    row['Source'] = data_source
    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)


def main(argv):
    parser = argparse.ArgumentParser(description='DDoS attacks detection with CNNs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-t', '--train', nargs='+', type=str, help='Start the training process')
    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int, help='Training epochs')
    parser.add_argument('-cv', '--cross_validation', default=5, type=int, help='Number of folds for cross-validation')
    parser.add_argument('-p', '--predict', nargs='?', const='.', default=None, type=str, help='Perform a prediction on pre-preprocessed data in a folder')
    parser.add_argument('-pl', '--predict_live', type=str, help='Perform a prediction on live traffic from an interface or pcap file')
    parser.add_argument('-i', '--iterations', default=1, type=int, help='Prediction iterations for performance measurement')
    parser.add_argument('-m', '--model', type=str, help='File containing the model for prediction')
    parser.add_argument('-a', '--attack_net', type=str, help='Subnet of the attacker (for labeling)')
    parser.add_argument('-v', '--victim_net', type=str, help='Subnet of the victim (for labeling)')
    parser.add_argument('-y', '--dataset_type', type=str, help='Type of dataset for labeling (e.g., DOS2019)')
    
    args = parser.parse_args()

    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    if args.train is not None:
        subfolders = glob.glob(os.path.join(args.train[0], "*", "")) or [args.train[0] + os.path.sep]
        
        for full_path in tqdm(subfolders, desc="Processing Folders"):
            dataset_folder = os.path.normpath(full_path)
            X_train, Y_train = load_dataset(os.path.join(dataset_folder, '*-train.hdf5'))
            X_val, Y_val = load_dataset(os.path.join(dataset_folder, '*-val.hdf5'))

            if X_train is None or Y_train is None:
                print(f"ERROR: Failed to load data from {dataset_folder}. Skipping.")
                continue

            X_train = X_train.astype('float32')
            Y_train = Y_train.astype('float32').reshape(-1, 1)
            X_val = X_val.astype('float32')
            Y_val = Y_val.astype('float32').reshape(-1, 1)

            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

            train_file = glob.glob(os.path.join(dataset_folder, '*-train.hdf5'))[0]
            filename = os.path.basename(train_file)
            parts = filename.split('-')
            time_window, max_flow_len, dataset_name = int(parts[0].replace('t','')), int(parts[1].replace('n','')), parts[2]

            print(f"\nTraining on dataset: {dataset_name}")

            model_name = f"{dataset_name}-LUCID"
            
            keras_classifier = KerasClassifier(
                model=Conv2DModel, model_name=model_name, input_shape=X_train.shape[1:], kernel_col=X_train.shape[2],
                learning_rate=0.01, kernels=64, regularization=None, dropout=None, weight_decay=0.0,dense_units=16, dense_dropout=0.25
            )
            
            rnd_search_cv = GridSearchCV(keras_classifier, HYPERPARAMETERS, cv=args.cross_validation, refit=True, n_jobs=-1, verbose=1)

            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
            best_model_filename = os.path.join(OUTPUT_FOLDER, f"{time_window}t-{max_flow_len}n-{model_name}")
            mc = tf.keras.callbacks.ModelCheckpoint(f"{best_model_filename}.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

            rnd_search_cv.fit(X_train, Y_train, epochs=args.epochs, validation_data=(X_val, Y_val), callbacks=[es, mc])

            best_model = rnd_search_cv.best_estimator_.model_
            best_model.save(f"{best_model_filename}.h5")

            Y_pred_val = (best_model.predict(X_val) > 0.5)
            f1_score_val = f1_score(Y_val, Y_pred_val)
            accuracy = accuracy_score(Y_val, Y_pred_val)

            print(f"\nBest Parameters: {rnd_search_cv.best_params_}")
            print(f"Best Model F1 Score on Validation Set: {f1_score_val:.4f}")


    if args.predict is not None:
        if not args.model:
            print("Error: Please specify a model file with -m or --model")
            return
        
        model = load_model(args.model)
        dataset_filelist = glob.glob(os.path.join(args.predict, '*test.hdf5'))
        
        predict_filepath = os.path.join(OUTPUT_FOLDER, f'predictions-{time.strftime("%Y%m%d-%H%M%S")}.csv')
        with open(predict_filepath, 'w', newline='') as predict_file:
            predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
            predict_writer.writeheader()

            for dataset_path in tqdm(dataset_filelist, desc="Predicting on test files"):
                X_test, Y_test = load_dataset(dataset_path)
                if X_test is None:
                    continue
                
                X_test = X_test.astype('float32')
                Y_test = Y_test.astype('float32').reshape(-1, 1)

                avg_time = 0
                Y_pred = None
                for _ in trange(args.iterations, desc="Iterations", leave=False):
                    start_time = time.time()
                    Y_pred = (model.predict(X_test, batch_size=2048) > 0.5).astype("int32")
                    end_time = time.time()
                    avg_time += (end_time - start_time)
                
                avg_time /= args.iterations
                packets = count_packets_in_dataset([X_test])[0]
                model_name = os.path.basename(args.model)

                report_results(np.squeeze(Y_test), np.squeeze(Y_pred), packets, model_name, os.path.basename(dataset_path), avg_time, predict_writer)


    if args.predict_live is not None:
        if not args.model:
            print("Error: Please specify a model file with -m or --model")
            return

        model = load_model(args.model)
        model_filename = os.path.basename(args.model)
        parts = model_filename.split('-')
        time_window, max_flow_len = int(parts[0].replace('t','')), int(parts[1].replace('n',''))
        
        if args.predict_live.endswith('.pcap'):
            cap = pyshark.FileCapture(args.predict_live)
        else:
            cap = pyshark.LiveCapture(interface=args.predict_live)
        
        labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)
        mins, maxs = static_min_max(time_window)

        print(f"Starting live prediction on {args.predict_live}...")
        for samples in process_live_traffic(cap, labels, max_flow_len, time_window):
            if not samples:
                continue

            X, Y_true, _ = dataset_to_list_of_fragments(samples)
            X_norm = normalize_and_padding(X, mins, maxs, max_flow_len)
            
            X_norm = np.array(X_norm).astype('float32')
            X_norm = np.reshape(X_norm, (X_norm.shape[0], X_norm.shape[1], X_norm.shape[2], 1))
            
            Y_pred = (model.predict(X_norm) > 0.5).astype("int32")


if __name__ == "__main__":
    main(sys.argv[1:])