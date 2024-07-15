import pandas as pd


def hold_out(df, train_proportion, seed=0):
    # df: dataframe pandas
    # train_proportion : 0<= train_proportion<=1 , quantita di dataset per training
    # seed: seed per shuffling
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_size = int(train_proportion * df.shape[0])

    df_train = shuffled_df.iloc[:train_size, :]
    df_test = shuffled_df.iloc[train_size:, :]

    return df_train, df_test


def kfold(df_design, fold_num=5, normalize=True, seed=0):
    shuffled_df = df_design.sample(frac=1, random_state=seed).reset_index(
        drop=True
    )

    assert df_design.shape[0] % fold_num == 0  # all folds are of equal size

    fold_len = df_design.shape[0] // fold_num

    fold_list = list()

    for i in range(fold_num):
        fold_list.append(
            shuffled_df.iloc[i * fold_len : (i + 1) * fold_len, :]  # noqa
        )

    fold_sets = []

    for fold in fold_list:
        other_folds = [f for f in fold_list if f is not fold]
        assert len(other_folds) == len(fold_list) - 1
        training_set = pd.concat(other_folds).reset_index(drop=True)

        assert (
            len(list(set(training_set["ID"]) & set(fold["ID"]))) == 0
        )  # checks any data leak

        fold_sets.append((training_set, fold))

    df_normal_parameters = pd.DataFrame()

    feature_cols = [f"x{i}" for i in range(1, 11)]  # feature column names
    # target_cols = [ f'y{i}' for i in range(1,4)]

    if normalize:
        for idx, (df_training, df_val) in enumerate(
            fold_sets
        ):  # normalize features

            # compute mean and std. dev.
            curr_means = df_training[feature_cols].mean()
            curr_std = df_training[feature_cols].std()

            # standardize training set
            df_training.loc[:, feature_cols] = (
                df_training[feature_cols] - curr_means
            ) / curr_std

            # standardize val set w.r.t mean and std dev of training
            df_val.loc[:, feature_cols] = (
                df_val[feature_cols] - curr_means
            ) / curr_std

            # saving the mean and std, since it is needed for assessment
            curr_means.rename(f"mean{idx}", inplace=True)
            curr_std.rename(f"std{idx}", inplace=True)
            df_normal_parameters = pd.concat(
                [df_normal_parameters, curr_means, curr_std], axis=1
            )

    return fold_sets
