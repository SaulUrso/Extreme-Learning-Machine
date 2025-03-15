# implementation of Nesterov accelerated gradient descent


import pandas as pd
import datautils
import modelutils as mu
import time
from sklearn.model_selection import ParameterGrid


DATASET = "Dataset/ML-CUP23-TR.csv"
df_cup = pd.read_csv(DATASET, skiprows=6)
df_cup.rename(columns={"# Training set: ID": "ID"}, inplace=True)

df_cup.info()

DESIGN_SIZE = 0.8
TRAIN_SIZE = 0.8

df_design, df_test = datautils.hold_out(df_cup, DESIGN_SIZE)
df_train, df_val = datautils.hold_out(df_design, TRAIN_SIZE)
X_train, y_train = datautils.obtain_features_targets(df_train)
X_val, y_val = datautils.obtain_features_targets(df_val)
X_test, y_test = datautils.obtain_features_targets(df_test)

input_size = X_train.shape[1]
output_size = y_train.shape[1]
hidden_size = 100
alpha = 1e-3
# iterate over multiple seeds

HIDDEN_SIZE = 1000
LEARNING_RATE = "auto"
BETA = 0
ALPHA = 1e-8
EPSILON = 1e-10
RESULTS = "Results/"

results_dict = {
    "Hidden size": [],
    "Alpha": [],
    "Seed": [],
    "Initialization type": [],
    "Train": [],
    "Validation": [],
    "Time": [],
}

hyperparameters = {
            "Hidden size": [50, 100, 1000, 2000],
            "Alpha": [1e-3, 1e-8],
            "Seed": [1, 2, 3, 4],
            "Initialization type": ["fan-in"],
            }

param_grid = list(ParameterGrid(hyperparameters))


for params in param_grid:
    # initialize the model
    elm = mu.ELM(
        input_size,
        params["Hidden size"],
        output_size,
        seed=params["Seed"],
        init=params["Initialization type"],
    )
    # measure the time
    print(params)
    start_time = time.process_time()

    # train model
    elm.computewoutsystem(X_train, y_train, alpha=params["Alpha"])

    train_pred = elm.predict(x=X_train)
    loss_train = mu.compute_loss(y_train, train_pred)

    val_pred = elm.predict(x=X_val)
    loss_val = mu.compute_loss(y_val, val_pred)

    end_time = time.process_time()

    # save the results

    results_dict["Hidden size"].append(params["Hidden size"])
    results_dict["Alpha"].append(params["Alpha"])
    results_dict["Seed"].append(params["Seed"])
    results_dict["Initialization type"].append(params["Initialization type"])
    results_dict["Train"].append(loss_train)
    results_dict["Validation"].append(loss_val)
    results_dict["Time"].append(end_time - start_time)

# save the results
results = pd.DataFrame(results_dict)
results.to_csv(
    RESULTS + "results_systemresolution.csv", index=False)
print("Results saved.")
