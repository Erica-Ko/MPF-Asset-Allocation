import data_layer_aia as data_layer
import strategy_tools as tool
import joblib
import pandas as pd
import os

### Load fund data ###
# output_dict = data_layer.main()
# contents = output_dict['Provider']['MPF Scheme Name']
# df = pd.read_json(contents)

with open('fund_data.txt', 'r') as file:
    contents = file.read()
df = pd.read_json(contents)

### Construct a df dict and work for features engineering ###
table_dict = {} # dict for storing df
for col in df.columns:
    table_dict[col] = pd.DataFrame(df[col])

### Loop through each df ###
for k,v in table_dict.items():
    fund_df = v.sort_index()
    fund_df = tool.get_hsi(fund_df)
    df_w_features = tool.feature_engineer(fund_df)
    
    ### Model training for that fund ###
    X_train, y_train, X_test, y_test = tool.train_test_split(df_w_features, 0.8)
    
    ### Check if the fund contains only one class ###
    if len(y_train.value_counts()) < 2:
        continue
        
    ### Build models ###
    log_re_model = tool.gen_logistic_regression(X_train, y_train, n_split=5)
    svm_model = tool.gen_SVM(X_train, y_train, n_split=5)
    lightgbm_model = tool.gen_binary_lightgbm(X_train, y_train, 50, 'precision')
    
    ### Save model ###
    current = os.getcwd()
    path = f'{current}/Models'
    filename = f'{path}/{k}.pickle'
    if not os.path.exists(path):
        os.mkdir(path)
    with open(filename, 'wb') as file:
        joblib.dump(log_re_model, file)
        joblib.dump(svm_model, file)
        joblib.dump(lightgbm_model.best_estimator_, file)