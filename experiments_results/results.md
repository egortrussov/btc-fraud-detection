# Results on both fixed and paper splits


| Dataset type | Split | Model | Precicion | Recall | Accuracy |
|----------|----------|----------|----------|----------|----------|
| Wallets + users  | Paper | RF | 0.9066 | 0.8069 | ?
| Wallets + users  | Paper | XGB | 0.8575 | 0.8032 | 0.9942
| Wallets + users + exchanges  | Paper | XGB | 0.8604 | 0.7997 | 0.9942
| Wallets + users  | Paper | XGB + users_heuristic | 0.7960 | **0.8194** | 0.9931
| Wallets only | Fixed | RF | 0.2378 | 0.1477 | ?
| Wallets only | Fixed | XGB | 0.3682 | 0.3829 | ?
| Wallets + users | Fixed | XGB | **0.4102** | **0.4277** | 0.9807
| Wallets + users + Exchanges | Fixed | XGB | **0.4314** | **0.4220** | 0.9807
| Wallets + users | Fixed | XGB + users_heuristic | 0.3959 | **0.5560** | 0.9800

(users heuristic - only for users with addresses number in (10 .. 1000))

## Feature importances

### New wallets features + users fields

| Feature | imp |
|----------|----------|
| outcoming_tx_cnt | 0.19234791 |
| output_address_cnt | 0.12838577 |
| btc_received_min | 0.096798696 |
| **outcoming_tx_input_address_cnt_mean** | 0.087779395 |
| **user_user_ts_fees_share_mean** | 0.052322146 |
| **incoming_tx_whole_fee_4_cnt** | 0.04154164 |
| outcoming_tx_ts_cnt | 0.038700536 |
| **user_btc_transacted_total** | 0.031234138 |
| **incoming_tx_ts_gini** | 0.026934415 |
| **user_active_time_steps_cnt** | 0.023902886 |
| **big_inp_addr** | 0.023800345 |
| **user_output_users_cnt** | 0.021842059 |
| input_address_cnt | 0.018661354 |
| ... | ... |
| user_incoming_tx_cnt | 0.0013532893 |
| user_user_ts_fees_share_min | 0.0011640645 |
| big_bct_received | 0.00079449534 |
| outcoming_tx_fees_total | 0.00035352886 |
| outcoming_tx_output_address_cnt_mean | 0.0 |
| user_btc_received_median | 0.0 |
| outcoming_tx_fees_min | 0.0 |
| outcoming_tx_whole_fee_4_cnt | 0.0 |
| outcoming_tx_ts_gini | 0.0 |
| user_whole_fee_5 | 0.0 |


## Confusion matrices

### Split from paper - XGBoost

|  | True 1 | True 0 |
|----------|----------|----------| 
| pred 1 | 3462 |	575 |	 
| pred 0	| 848 |	241998	|


### Split from paper - XGBoost + users heuristic

|  | True 1 | True 0 |
|----------|----------|----------| 
| pred 1 | 3532 |	905 |	 
| pred 0	| 778 |	241668	|


### Correct split - XGBoost

|  | True 1 | True 0 |
|----------|----------|----------| 
| pred 1 | 2124 |	3517 |	 
| pred 0	| 2381 |	273012	|


### Correct split - XGBoost + users heuristic

|  | True 1 | True 0 |
|----------|----------|----------| 
| pred 1 | 2505 |	3822 |	 
| pred 0	| 2000 |	272707	|
