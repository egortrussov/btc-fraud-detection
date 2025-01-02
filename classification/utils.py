from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
    best_ind = np.argmax(f1_hist)

    if plot:
        print(f"""
        Best threshold: {thrs[best_ind]}
        Best F1: {f1_hist[best_ind]}, precicion: {precicion_hist[best_ind]}, recall: {recall_hist[best_ind]}
        """)
            
        fig, ax = plt.subplots(1,3,figsize=(12,3))
        sns.lineplot(x=thrs, y=precicion_hist, ax=ax[0])
        sns.lineplot(x=thrs, y=recall_hist, ax=ax[1])
        sns.lineplot(x=thrs, y=f1_hist, ax=ax[2])
        
        ax[0].set_title("Precicion")
        ax[1].set_title("Recall")
        ax[2].set_title("F1")
    
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
