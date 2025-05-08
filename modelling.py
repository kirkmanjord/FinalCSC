from tensorflow.keras.layers import Input, LSTM, Concatenate, Dense, Masking
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import pickle as pkl
import numpy as np
import matplotlib
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras import Sequential
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs Available:", tf.config.list_physical_devices('GPU'))


def evaluate_ensemble_calibration(models, X, y_true, num_bins=10):
    """
    For ensemble sizes 1..len(models):
      • averages predictions of the first i models
      • computes Expected Calibration Error (ECE) with num_bins
      • prints ECE(i)
    Finally plots ECE vs. ensemble size.

    Args:
        models   : list of fitted Keras models
        X        : input data (array or list of arrays for multi-input)
        y_true   : true binary labels (shape (N,) or (N,1))
        num_bins : number of bins to use for ECE
    """
    # make sure y_true is 1D
    y_true = np.array(y_true).flatten()
    N = len(y_true)
    ece_vals = []

    for i in range(1, len(models) + 1):
        # ---- ensemble predictions ----
        # sum preds of first i models
        y_pred_sum = None
        for m in models[:i]:
            p = m.predict(X)
            if y_pred_sum is None:
                y_pred_sum = p
            else:
                y_pred_sum += p
        # average & flatten to 1D
        y_pred = (y_pred_sum / i).flatten()

        # ---- compute ECE ----
        bin_edges = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        for b in range(num_bins):
            mask = (y_pred >= bin_edges[b]) & (y_pred < bin_edges[b + 1])
            bin_count = mask.sum()
            if bin_count > 0:
                acc_bin = y_true[mask].mean()
                conf_bin = y_pred[mask].mean()
                ece += (bin_count / N) * abs(acc_bin - conf_bin)

        ece_vals.append(ece)
        print(f"Ensemble size {i:2d}: ECE = {ece:.4f}")

    # ---- plot ECE progression ----
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(models) + 1), ece_vals, marker='o')
    plt.xlabel('Number of models in ensemble')
    plt.ylabel('Expected Calibration Error')
    plt.title('Ensemble Calibration Error vs. Ensemble Size')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class ExpectedCalibrationError(tf.keras.metrics.Metric):
    def __init__(self, num_bins=10, name='ece', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_bins = num_bins
        # Bin boundaries [0, 1/num_bins, 2/num_bins, ..., 1]
        boundaries = [i / num_bins for i in range(num_bins + 1)]
        self.bin_lowers = tf.constant(boundaries[:-1], dtype=tf.float32)
        self.bin_uppers = tf.constant(boundaries[1:], dtype=tf.float32)

        # Running sums
        self.bin_counts = self.add_weight(name='bin_counts',
                                          shape=(num_bins,),
                                          initializer='zeros')
        self.acc_sums = self.add_weight(name='acc_sums',
                                        shape=(num_bins,),
                                        initializer='zeros')
        self.conf_sums = self.add_weight(name='conf_sums',
                                         shape=(num_bins,),
                                         initializer='zeros')
        self.total = self.add_weight(name='total_samples',
                                     initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: either int labels or one-hot
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        # If multi‑class, take max confidence and predicted class
        if y_pred.shape[-1] > 1:
            confidences = tf.reduce_max(y_pred, axis=-1)
            preds = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        else:
            confidences = tf.reshape(y_pred, [-1])
            preds = tf.cast(confidences > 0.5, tf.int32)

        # accuracy per sample
        accuracies = tf.cast(tf.equal(preds, y_true), tf.float32)

        # Shape (batch, 1) to compare to each bin
        conf_exp = tf.reshape(confidences, [-1, 1])
        # Compare to each bin: shape (batch, num_bins)
        in_lower = tf.greater_equal(conf_exp, self.bin_lowers)
        in_upper = tf.less(conf_exp, self.bin_uppers)
        in_bin = tf.logical_and(in_lower, in_upper)

        # Counts per bin this batch
        counts_batch = tf.reduce_sum(tf.cast(in_bin, tf.float32), axis=0)
        acc_sums_batch = tf.reduce_sum(tf.cast(in_bin, tf.float32) * tf.reshape(accuracies, [-1, 1]),
                                       axis=0)
        conf_sums_batch = tf.reduce_sum(tf.cast(in_bin, tf.float32) * tf.reshape(confidences, [-1, 1]),
                                        axis=0)
        batch_size = tf.cast(tf.shape(confidences)[0], tf.float32)

        # Update running totals
        self.bin_counts.assign_add(counts_batch)
        self.acc_sums.assign_add(acc_sums_batch)
        self.conf_sums.assign_add(conf_sums_batch)
        self.total.assign_add(batch_size)

    def result(self):
        # Avoid divide‑by‑zero
        counts = self.bin_counts
        acc_bin = tf.math.divide_no_nan(self.acc_sums, counts)
        conf_bin = tf.math.divide_no_nan(self.conf_sums, counts)
        prop_bin = tf.math.divide_no_nan(counts, self.total)

        # ECE = sum_i  p(i) * | acc(i) - conf(i) |
        ece = tf.reduce_sum(prop_bin * tf.abs(acc_bin - conf_bin))
        return ece

    def reset_states(self):
        for v in [self.bin_counts, self.acc_sums, self.conf_sums, self.total]:
            v.assign(tf.zeros_like(v))


with open('teamDict.pkl', 'rb') as f:
    loaded_dict = pkl.load(f)

val = loaded_dict.pop('SuperSonics')
val = loaded_dict.pop('Bobcats')

val = loaded_dict.pop('Thunder')
val = loaded_dict.pop('Pelicans')


def createBatchFromDict(teamDict, numGame):
    team1Data = []
    team2Data = []
    truths = []
    gameIDs = []
    for i in range(0, 1800 - numGame):
        for teamName in teamDict.keys():
            #get the sequence of the desired team
            sequence = teamDict[teamName]
            #get the sequence of the opponent
            opTeam = sequence.loc[i + numGame, 'nameOfOpponent']
            if opTeam in teamDict.keys():
                opSequence = teamDict[opTeam]
                gameIDs.append(sequence.loc[numGame + i, sequence.columns.str.contains('gameId')].iloc[0])
                columns_to_remove = ['Id', 'Date', 'nameOfOpponent', 'win', 'Minutes']
                pattern = '|'.join(columns_to_remove)
                truth = sequence.loc[numGame + i, sequence.columns.str.contains('win')].iloc[0]

                sequence = sequence.loc[i:numGame + i - 1, ~sequence.columns.str.contains(pattern)]
                opSequence = opSequence.loc[i:numGame + i - 1, ~opSequence.columns.str.contains(pattern)]
                truths.append(truth)
                team1Data.append(sequence.to_numpy())
                #colp = sequence.iloc[:,165:166].columns
                team2Data.append(opSequence.to_numpy())

        print('hi')
    X1 = np.stack(team1Data)
    X2 = np.stack(team2Data)
    return X1, X2, truths, gameIDs

def importmore():
    team1Data, team2Data, truths,gameIds=createBatchFromDict(loaded_dict, 20)
    # 2. Pickle (serialize) to a file
    with open('team1Data.pkl', 'wb') as f:
        pickle.dump(team1Data, f)
    with open('team2Data.pkl', 'wb') as f:
        pickle.dump(team2Data, f)
    # 3. Later—load it back
    with open('truths.pkl', 'wb') as f:
        pickle.dump(truths, f)
    with open('gameIds.pkl', 'wb') as f:
        pickle.dump(gameIds, f)
    # 3. Later—load it back

with open('team1Data.pkl', 'rb') as f:
    team1Data = pickle.load(f)
with open('team2Data.pkl', 'rb') as f:
    team2Data = pickle.load(f)
with open('truths.pkl', 'rb') as f:
    truths = pickle.load(f)
with open('gameIds.pkl', 'rb') as f:
    gameIds = pickle.load(f)
# Parameters
indices = [i for i, val in enumerate(gameIds) if val == 22101044]
input_dim = team1Data.shape[2]
print(f'd{input_dim}')
ensemble = []
X1_tr, X1_val, X2_tr, X2_val, y_tr, y_val = train_test_split(
    team1Data, team2Data, np.array(truths),
    test_size=0.2, shuffle=True, random_state=42
)
for i in range(4):
    lstm_units = 50
    drop_rate = 0.3  # you can tweak this
    indices = np.random.choice(X1_tr.shape[0], X1_tr.shape[0], replace=True)
    curX1 = X1_tr[indices]
    curX2 = X2_tr[indices]
    curY = y_tr[indices]
    # inputs
    input1 = Input(shape=(None, input_dim), name=f'input1{i}')
    input2 = Input(shape=(None, input_dim), name=f'input2{i}')

    # optional masking if you have variable‑length with padding
    # input1m = Masking()(input1)
    # input2m = Masking()(input2)
    # → then feed input1m / input2m into the LSTM
    l2_strength = 1e-5  # Stronger than default (1e-4)
    # shared LSTM with internal dropout
    ece_metric = ExpectedCalibrationError(num_bins=10)
    shared_lstm = LSTM(
        lstm_units,
        dropout=drop_rate,  # drop inputs to each LSTM cell
        recurrent_dropout=drop_rate,  # drop the recurrent state
        kernel_regularizer=regularizers.l2(l2_strength),
        recurrent_regularizer=regularizers.l2(l2_strength),
        bias_regularizer=regularizers.l2(l2_strength),
        name=f'shared_lstm{i}'
    )

    lstm_out1 = shared_lstm(input1)
    lstm_out2 = shared_lstm(input2)

    # merge and apply Dropout again
    merged = Concatenate(name=f'concatenate{i}')([lstm_out1, lstm_out2])
    x = Dropout(0.3, name=f'dropout_after_lstm{i}')(merged)

    # (optional) add a small hidden layer before final output
    # x = Dense(16, activation='relu', name='hidden')(x)
    # x = Dropout(0.3, name='dropout_hidden')(x)

    output = Dense(1, activation='sigmoid', name=f'logistic_output{i}')(x)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=[ece_metric, 'accuracy'])

    # Fit the model and capture the training history

    loss, acc, acc2 = model.evaluate([X1_val, X2_val], y_val, verbose=0)
    history = model.fit(
        [curX1, curX2], curY,
        validation_data=([X1_val, X2_val], y_val),
        shuffle=True,
        epochs=5,
        batch_size=50,
    )
    ensemble.append(model)
def notrn():
    with open('ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    with open('truths.pkl', 'rb') as f:
        ensemble = pickle.load(f)

def plot_calibration_curve(models, X, y_true, n_bins=10):
    # Step 1: Predict probabilities

    y_pred = None
    for i in range(len(models)):
        if y_pred is None:
            y_pred = models[i].predict(X)
        else:
            y_pred += models[i].predict(X)
    y_pred = (y_pred / len(models)).flatten()
    y_true = np.array(y_true).flatten()

    # Step 2: Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_true = []
    bin_pred = []

    # Step 3: Loop over bins
    for i in range(n_bins):
        # Get mask for current bin
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_true.append(np.mean(y_true[mask]))
            bin_pred.append(np.mean(y_pred[mask]))
        else:
            bin_true.append(np.nan)
            bin_pred.append(np.nan)

    # Step 4: Plot
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(bin_pred, bin_true, 'o-', label='Model calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def runEnsemble(models, X):
    y_pred = None
    for i in range(len(models)):
        if y_pred is None:
            y_pred = models[i].predict(X)
        else:
            y_pred += models[i].predict(X)
    y_pred = (y_pred / len(models)).flatten()
    return y_pred

def bin_mean_map(preds, actuals, num_bins=20, value_range=(0.0, 1.0)):
    """
    For each element in `preds`, find its bin and return the
    average of `actuals` in that same bin.

    Args:
        preds     : 1D array-like of predictions (shape (N,))
        actuals   : 1D array-like of true labels or win rates (shape (N,))
        num_bins  : how many equal-width bins to split [min,max] into
        value_range: (min, max) range to bin the preds over

    Returns:
        mapped    : np.ndarray of shape (N,), where
                    mapped[i] = mean(actuals[j] for all j in the same bin as preds[i])
        bin_edges : np.ndarray of shape (num_bins+1,) the bin boundaries
    """
    # Flatten inputs
    preds = np.asarray(preds).flatten()
    actuals = np.asarray(actuals).flatten()
    N = len(preds)
    assert actuals.shape[0] == N, "preds and actuals must be same length"

    # 1. Compute bin edges
    bin_edges = np.linspace(value_range[0], value_range[1], num_bins + 1)

    # 2. Digitize preds into bins [0..num_bins-1]
    bin_idx = np.digitize(preds, bin_edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    # 3. Compute per-bin mean of actuals
    mapped = np.empty(N, dtype=float)
    for b in range(num_bins):
        mask = (bin_idx == b)
        if np.any(mask):
            mapped[mask] = actuals[mask].mean()
        else:
            mapped[mask] = np.nan  # or choose bin midpoint

    return mapped
def create_logistic_regression(input_dim, lr=1e-3):

    model = Sequential([
        Dense(5,
              activation='relu',
              input_shape=(input_dim,),
              name='hidden_layer'),
        Dense(1,
              activation='sigmoid',
              name='logistic_output')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mse']
    )
    return model

def plot_calibration_curve_with_calibration(ensemble, calibration,X, y_true, n_bins=10):
    y_pred = runEnsemble(ensemble, X)
    y_pred = calibration.predict(y_pred).flatten()
    # Step 2: Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_true = []
    bin_pred = []

    # Step 3: Loop over bins
    for i in range(n_bins):
        # Get mask for current bin
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_true.append(np.mean(y_true[mask]))
            bin_pred.append(np.mean(y_pred[mask]))
        else:
            bin_true.append(np.nan)
            bin_pred.append(np.nan)

    # Step 4: Plot
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(bin_pred, bin_true, 'o-', label='Model calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve with Calibrated Model')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





evaluate_ensemble_calibration(ensemble, [X1_val, X2_val], y_val, num_bins=10)


calibrationModel = create_logistic_regression(input_dim=1)
fitter = bin_mean_map(runEnsemble(ensemble,[X1_val, X2_val]), y_val)
calibrationModel.fit(runEnsemble(ensemble,[X1_tr, X2_tr]),bin_mean_map(runEnsemble(ensemble,[X1_tr, X2_tr]),y_tr), epochs=1, batch_size=32)

def evaluate_calibration(y_pred, y_true, num_bins=20):
    """
    Compute the Expected Calibration Error (ECE).

    Args:
        y_pred   : 1D array of predicted probabilities (floats in [0,1]).
        y_true   : 1D array of true binary labels (0 or 1).
        num_bins : number of equal-width bins to use.

    Returns:
        ece      : the scalar ECE value.
    """
    # 1. Flatten inputs
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    assert y_pred.shape == y_true.shape, "predictions and labels must match length"
    N = y_pred.shape[0]

    # 2. Define bin edges and initialize ECE
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    # 3. Loop over bins, accumulate weighted |acc - conf|
    for i in range(num_bins):
        # lower and upper edge of bin i
        lo, hi = bin_edges[i], bin_edges[i+1]
        # mask of samples in bin
        mask = (y_pred >= lo) & (y_pred < hi)
        bin_count = mask.sum()

        if bin_count > 0:
            # average true label (accuracy) in this bin
            acc_bin  = y_true[mask].mean()
            # average predicted prob (confidence) in this bin
            conf_bin = y_pred[mask].mean()
            # weight by bin fraction
            ece += (bin_count / N) * abs(acc_bin - conf_bin)

    return ece
print(evaluate_calibration(runEnsemble(ensemble,[X1_tr, X2_tr] ), y_tr))
print(evaluate_calibration(calibrationModel.predict(runEnsemble(ensemble,[X1_tr, X2_tr] )), y_tr))
plot_calibration_curve(ensemble, [X1_val, X2_val], y_val, n_bins=10)
#plot_calibration_curve_with_calibration(ensemble, calibrationModel,[X1_val, X2_val], y_val, n_bins=10)

# Plot the training and validation accuracy
print(f"The probability:{calibrationModel.predict(runEnsemble(ensemble, [team1Data[45432:45433], team2Data[45432:45433]]))} ")






#type space kirkman Jordan j kirkmankrikman krikmankirkman kirkman  sd   fdjj    k
