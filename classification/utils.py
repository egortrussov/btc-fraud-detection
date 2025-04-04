from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

np.random.seed(0)


COMMON_FEATURES = [
    # 'addrId',
    'class',
    'num_timesteps_appeared_in', 'num_addr_transacted_multiple',
    'num_txs_as_sender',
    'num_txs_as receiver', 'lifetime_in_blocks', 'total_txs',
    'btc_transacted_total', 'btc_sent_total', 'btc_received_total',
    'fees_total', 'transacted_w_address_total', 'first_block_appeared_in',
    'first_sent_block', 'first_received_block', 'btc_transacted_min',
    'btc_sent_min', 'btc_received_min', 'fees_min',
    'transacted_w_address_min', 'btc_transacted_max', 'btc_sent_max',
    'btc_received_max', 'fees_max', 'last_block_appeared_in',
    "fees_median",
    'transacted_w_address_max'
]

USERS_FEATURES = [
    'user_ts_fees_share_mean', 'user_ts_fees_share_min',
    'user_ts_fees_share_max', 'user_addr_cnt', 'user_outcoming_tx_cnt',
    'user_incoming_tx_cnt', 'user_input_users_cnt', 'user_output_users_cnt',
    'user_active_time_steps_cnt', 'user_btc_sent_total',
    'user_btc_received_total', 'user_interracted_output_address_cnt',
    'user_interracted_input_address_cnt', 'user_overall_activity_coef',
    'user_whole_fee_5', 
]

CUSTOM_FEATURES = [
    'addr_gini', 'whole_fees_5', 
]

def prepare_wallets_features_data(data, type):
    if type == "raw":
        return data[COMMON_FEATURES]
    if type == "with_users":
        return data[COMMON_FEATURES + USERS_FEATURES]
    if type == "full":
        return data[COMMON_FEATURES + USERS_FEATURES + CUSTOM_FEATURES]
    if type == "full_with_extra_flags":
        res = data[COMMON_FEATURES + USERS_FEATURES]
        
        res["one_ts_active"] = (res["num_timesteps_appeared_in"] == 1).astype(int)
        res["user_one_ts_active"] = (res["user_active_time_steps_cnt"] == 1).astype(int)

        res["user_one_addr_flg"] = (res["user_addr_cnt"] == 1).astype(int)

        res["no_sending"] = (res["user_outcoming_tx_cnt"] == 0).astype(int)
        res["one_input_tx"] = (res["num_txs_as_sender"] == 1).astype(int)
        res["one_output_tx"] = (res["num_txs_as receiver"] == 1).astype(int)

        # res1 = res[res["num_txs_as receiver"] == 0]
        # res2 = res[res["num_txs_as receiver"] != 0]
        # res2["input_to_output_tx_ratio"] = (
        #     res2["num_txs_as_sender"] / res2["num_txs_as receiver"]
        # ).fillna(0)
        # res1["input_to_output_tx_ratio"] = 0
        # res = pd.concat([res1, res2])

        # res["one_receiving"] = (res["user_incoming_tx_cnt"] == 1).astype(int)
        # res["one_sending"] = (res["user_outcoming_tx_cnt"] == 1).astype(int)
        # res["one_sending_and_receiving"] = ((res["user_outcoming_tx_cnt"] == 1) & (res["user_incoming_tx_cnt"] == 1)).astype(int)
        # res["no_receiving"] = (res["num_txs_as receiver"] == 0).astype(int)
        # res["many_addr_nei"] = (res["transacted_w_address_total"] >= 300)
        # res["many_input_btc"] = (res["btc_received_total"] >= np.quantile(res["btc_received_total"], 0.95))
        
        # res["one_output_and_input_tx"] = ((res["num_txs_as_sender"] == 1) & (res["num_txs_as receiver"] == 1)).astype(int)
        
        
        return res
    raise NotImplementedError

def prepare_wallets_features_data_new(data, type):
    return data.drop([
        "userId",
        "Time step",
        "addrId",
    ], axis=1)
    if type == "raw":
        return data[COMMON_FEATURES]
    if type == "with_users":
        return data[COMMON_FEATURES + USERS_FEATURES]
    if type == "full":
        return data[COMMON_FEATURES + USERS_FEATURES + CUSTOM_FEATURES]
    if type == "full_with_extra_flags":
        res = data[COMMON_FEATURES + USERS_FEATURES]
        
        res["one_ts_active"] = (res["num_timesteps_appeared_in"] == 1).astype(int)
        res["user_one_ts_active"] = (res["user_active_time_steps_cnt"] == 1).astype(int)

        res["user_one_addr_flg"] = (res["user_addr_cnt"] == 1).astype(int)

        res["no_sending"] = (res["user_outcoming_tx_cnt"] == 0).astype(int)
        res["one_input_tx"] = (res["num_txs_as_sender"] == 1).astype(int)
        res["one_output_tx"] = (res["num_txs_as receiver"] == 1).astype(int)

        # res1 = res[res["num_txs_as receiver"] == 0]
        # res2 = res[res["num_txs_as receiver"] != 0]
        # res2["input_to_output_tx_ratio"] = (
        #     res2["num_txs_as_sender"] / res2["num_txs_as receiver"]
        # ).fillna(0)
        # res1["input_to_output_tx_ratio"] = 0
        # res = pd.concat([res1, res2])

        # res["one_receiving"] = (res["user_incoming_tx_cnt"] == 1).astype(int)
        # res["one_sending"] = (res["user_outcoming_tx_cnt"] == 1).astype(int)
        # res["one_sending_and_receiving"] = ((res["user_outcoming_tx_cnt"] == 1) & (res["user_incoming_tx_cnt"] == 1)).astype(int)
        # res["no_receiving"] = (res["num_txs_as receiver"] == 0).astype(int)
        # res["many_addr_nei"] = (res["transacted_w_address_total"] >= 300)
        # res["many_input_btc"] = (res["btc_received_total"] >= np.quantile(res["btc_received_total"], 0.95))
        
        # res["one_output_and_input_tx"] = ((res["num_txs_as_sender"] == 1) & (res["num_txs_as receiver"] == 1)).astype(int)
        
        
        return res
    raise NotImplementedError


def get_training_data(data_raw, no_unknown=True, binary=True):
    if no_unknown:
        data = data_raw[data_raw["class"] != 3]
    else:
        data = data_raw
    
    X = data.drop("class", axis=1)

    if binary:
        y = (data["class"] == 1).astype(int)
    else:
        y = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def get_best_threshold(pred_probas, y_true, plot=True, thrs_cnt=20, return_hists=False):
    thrs = np.linspace(0.001, 0.999, thrs_cnt)
    precicion_hist = []
    recall_hist = []
    f1_hist = []
    fb_hist = []
    for thr in thrs:
        y_pred = (pred_probas >= thr).astype(int)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        if prec + rec == 0:
            f1 = 0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        precicion_hist.append(prec)
        recall_hist.append(rec)
        f1_hist.append(f1)
        fb_hist.append(fbeta_score(y_true, y_pred, beta=4))
    best_ind = np.argmax(fb_hist)

    if plot:
        print(f"""
        Best threshold: {thrs[best_ind]}
        Best F1: {f1_hist[best_ind]}, precicion: {precicion_hist[best_ind]}, recall: {recall_hist[best_ind]}
        """)
            
        fig, ax = plt.subplots(1,4,figsize=(15,3))
        sns.lineplot(x=thrs, y=precicion_hist, ax=ax[0])
        sns.lineplot(x=thrs, y=recall_hist, ax=ax[1])
        sns.lineplot(x=thrs, y=f1_hist, ax=ax[2])
        sns.lineplot(x=thrs, y=fb_hist, ax=ax[3])
        
        ax[0].set_title("Precicion")
        ax[1].set_title("Recall")
        ax[2].set_title("F1")
        ax[3].set_title("F-beta")
    
    if return_hists:
        return {
            "thr": thrs[best_ind],
            "precicion": precicion_hist[best_ind],
            "recall": recall_hist[best_ind],
            "f1": f1_hist[best_ind],
        }, thrs, precicion_hist, recall_hist
    return {
        "thr": thrs[best_ind],
        "precicion": precicion_hist[best_ind],
        "recall": recall_hist[best_ind],
        "f1": f1_hist[best_ind],
    }


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

def get_addr_position(addrId, wallets_data):
    wallets_addrs = wallets_data["addrId"]
    classes = wallets_data["class"]
    train, test, _, _ = train_test_split(wallets_addrs, classes, test_size=0.3, random_state=42)
    test = test.reset_index()["addrId"]
    train = train.reset_index()["addrId"]
    if train.isin([addrId]).sum():
        return "train", train.index[train == addrId].tolist()[0]
    if test.isin([addrId]).sum():
        return "test", test.index[test == addrId].tolist()[0]
    return None


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


def get_feature_to_importance(dataset_cols, feature_imps):
    for ind in np.argsort(feature_imps)[::-1]:
        print(dataset_cols[ind], feature_imps[ind])


def get_best_metric_for_optuna(y_pred, y_true, interval=(0.2, 0.4)):
    thrs = np.linspace(interval[0], interval[1], 10)
    metrics_hist = []

    for thr in thrs:
        metrics_hist.append( fbeta_score(y_true, (y_pred >= thr).astype(int), beta=2) )
    return max(metrics_hist), thrs[np.array(metrics_hist).argmax()]


def get_not_found_fraud_wallets(pred_classes, real_classes):
    pass
