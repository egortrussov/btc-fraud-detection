from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

np.random.seed(0)


def add_flags(df):
    df["big_bct_received"] =( df["btc_received_total"] > 10).astype(int)
    df["big_inp_addr"] =( df["input_address_cnt"] > 20).astype(int)

    return df

def get_prepared_train_test(raw_train, raw_test, type="full"):
    """
    type: (full, no_user)
    """
    if type == "full":
        fields_to_exc = [
            "addrId",
            "Time step",
            "first_block_appeared_in",
            "last_block_appeared_in",

            "incoming_tx_fees_mean",

            "userId"
        ]
    elif type == "no_users":
        users_fields = list(filter( lambda x : "user" in x, list(raw_train.columns)))
        fields_to_exc = [
            "addrId",
            "Time step",
            "first_block_appeared_in",
            "last_block_appeared_in",
            "incoming_tx_fees_mean",

            *users_fields

        ]
    else:
        raise Exception("Unknown dataset type")

    raw_train = raw_train.drop(fields_to_exc, axis=1)
    raw_test = raw_test.drop(fields_to_exc, axis=1)
    
    raw_train_prep = add_flags(raw_train)
    raw_test_prep = add_flags(raw_test)
    
    x_train = raw_train_prep.drop("class", axis=1)
    y_train = (raw_train_prep["class"] == 1).astype(int)

    x_test = raw_test_prep.drop("class", axis=1)
    y_test = (raw_test_prep["class"] == 1).astype(int)



    return x_train.reset_index().drop("index", axis=1), \
        x_test.reset_index().drop("index", axis=1), \
        y_train.reset_index().drop("index", axis=1)["class"], \
        y_test.reset_index().drop("index", axis=1)["class"]

def print_confusion_matrix(pred_proba, thr, y_true, with_prec_rec=False):
    y_pred = (pred_proba >= thr).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    print(f"\treal 1\treal 0")
    print(f"pred 1\t{tp}\t{fp}\t")
    print(f"pred 0\t{fn}\t{tn}\t")
    if with_prec_rec:
        print(f"Precicion {tp / (tp + fp)}, Recall {tp / (tp + fn)}, ")

def get_anomaly_detection_report(
        licit_labels,
        illicit_labels,
        unknown_labels,
):
    print(f"Fraud wallets found pct: {(illicit_labels == -1).sum() / illicit_labels.shape[0]}")

    print(f"Licit wallets fraud pct: {(licit_labels == -1).sum() / licit_labels.shape[0]}")
    print(f"Licit wallets accuracy: {(licit_labels == 1).sum() / licit_labels.shape[0]}")

    print(f"Unknown wallets fraud pct: {(unknown_labels == -1).sum() / unknown_labels.shape[0]}")
    print(f"Unknown wallets accuracy: {(unknown_labels == 1).sum() / unknown_labels.shape[0]}")

    print("")

    all_preds = np.concatenate([
        licit_labels == -1,
        illicit_labels == -1,
        unknown_labels == -1,
    ]).astype(int)
    all_true = np.concatenate([
        np.zeros(licit_labels.shape),
        np.ones(illicit_labels.shape),
        np.zeros(unknown_labels.shape),  
    ])

    
    print(f"Total accuracy: {accuracy_score(all_true, all_preds)}")
    print(f"Total precicion: {precision_score(all_true, all_preds)}")
    print(f"Total recall: {recall_score(all_true, all_preds)}")

    print(f"Total illicit cnt: {(all_preds == 1).sum()} / {all_preds.shape[0]}, true illicit: {illicit_labels.shape[0]}")

def get_detection_score(
        licit_labels,
        illicit_labels,
        unknown_labels,
):
    all_preds = np.concatenate([
        licit_labels == -1,
        illicit_labels == -1,
        unknown_labels == -1,
    ]).astype(int)
    all_true = np.concatenate([
        np.zeros(licit_labels.shape),
        np.ones(illicit_labels.shape),
        np.zeros(unknown_labels.shape),  
    ])

    return fbeta_score(all_true, all_preds, beta=3), precision_score(all_true, all_preds), recall_score(all_true, all_preds)


def get_best_score_threshold(
        licit_scores,
        illicit_scores,
        unknown_scores,
        plot=True,
        thr_interval=(-1,1),
        thrs_cnt=10
):
    all_true = np.concatenate([
        np.zeros(licit_scores.shape),
        np.ones(illicit_scores.shape),
        np.zeros(unknown_scores.shape),  
    ])

    precicion_hist = []
    recall_hist = []
    fb_hist = []

    thrs = np.linspace(thr_interval[0], thr_interval[1], 40)

    for thr in thrs:
        all_preds = np.concatenate([
            licit_scores >= thr,
            illicit_scores >= thr,
            unknown_scores >= thr,
        ]).astype(int)

        prec = precision_score(all_true, all_preds)
        rec = recall_score(all_true, all_preds)
        fb = fbeta_score(all_true, all_preds, beta=3)

        precicion_hist.append(prec)
        recall_hist.append(rec)
        fb_hist.append(fb)

    if plot:
        best_thr = thrs[np.array(fb_hist).argmax()]

        # best_thr = 0.05
        best_thr = 0.02
        all_preds_best = np.concatenate([
            licit_scores >= best_thr,
            illicit_scores >= best_thr,
            unknown_scores >= best_thr,
        ]).astype(int)

        print_confusion_matrix(all_preds_best, 0.1, all_true)

        fig, ax = plt.subplots(1,3,figsize=(15,3))
        print(precicion_hist)
        sns.lineplot(x=thrs, y=precicion_hist, ax=ax[0])
        sns.lineplot(x=thrs, y=recall_hist, ax=ax[1])
        sns.lineplot(x=thrs, y=fb_hist, ax=ax[2])
        
        ax[0].set_title("Precicion")
        ax[1].set_title("Recall")
        ax[2].set_title("F-beta")

        plt.show()
