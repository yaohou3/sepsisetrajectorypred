import pandas as pd
import tensorflow as tf
import numpy as np
import random

from datetime import datetime

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

print(tf.__version__)

tf.keras.utils.set_random_seed(292)

# set the number of rows used in each prediction (e.g. 32 rows is 32 / 4 = 8 hours of data).
predict_rows = 96

# set the last session id (e.g. 64 is 64 / 4 = 16 hours after admission
last_session = 144

dat = pd.read_csv('.../data.csv')

# Get unique IDs for each condition
ids_clust3_train1 = dat.loc[(dat['clust'] == 3) & (dat['train'] == 1), 'id'].unique()
ids_clust2_train1 = dat.loc[(dat['clust'] == 2) & (dat['train'] == 1), 'id'].unique()

# Randomly sample of IDs for each condition
np.random.seed(292)  

sampled_ids_clust3_train1 = np.random.choice(
    ids_clust3_train1, size=int(0.2 * len(ids_clust3_train1)), replace=False
)
sampled_ids_clust2_train1 = np.random.choice(
    ids_clust2_train1, size=int(0.4 * len(ids_clust2_train1)), replace=False
)

# Create a filter to keep:
# - Rows where `id` is in the sampled list for clust == 3 and train == 1
# - Rows where `id` is in the sampled list for clust == 2 and train == 1
# - All rows for other IDs
dat_filtered = dat[
    ((dat['id'].isin(sampled_ids_clust3_train1)) & (dat['clust'] == 3) & (dat['train'] == 1)) |
    ((dat['id'].isin(sampled_ids_clust2_train1)) & (dat['clust'] == 2) & (dat['train'] == 1)) |
    (~dat['id'].isin(np.concatenate([ids_clust3_train1, ids_clust2_train1])))
]

dat=dat_filtered

# Replace the unique IDs with sequential numbers from 1 to the total number of unique IDs
unique_ids = dat['id'].unique()
id_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_ids, start=1)}
dat['id'] = dat['id'].map(id_mapping)

# data frame with unique admission ids and train flag
ids = dat.filter(["id","train","clust"]).drop_duplicates()
ids = ids.sort_values(by = ["id"])

# serial data
dat = dat.sort_values(by = ["id","row_id"])
dat = dat.drop(["dttm","train","prlos_death","clust","intinf","pulm_edema","pleural_eff","lptt"], axis=1)

dat_long = np.array([(x,y,z) for x in range(ids.shape[0]) for y in np.linspace(start=8, stop=last_session, num=int(last_session/8)) for z in range(-1*(predict_rows-8),last_session+1)])
dat_long = pd.DataFrame(dat_long, columns = ['id','session_id','row_id']) # rename
dat_long = dat_long.assign(predict_rows = predict_rows).astype('float64')

# create index of rows to keep, i.e. row id less than session id and within predict rows prior to session id
indx = (dat_long.row_id <= dat_long.session_id) & (dat_long.row_id > dat_long.session_id - dat_long.predict_rows)

dat_long = dat_long[(indx)].drop('predict_rows', axis=1)

# drop any session_ids that are greater than the last maximum row_id
tmp = dat.sort_values(by = ["id","row_id"]).groupby(["id"]).tail(n=1)
tmp["max_row_id"] = tmp["row_id"]
tmp = tmp.filter(["id","max_row_id"])

dat_long = pd.merge(left=dat_long, right=tmp, on=["id"], how='inner')

dat_long = dat_long[(dat_long["session_id"] <= dat_long["max_row_id"])].drop(["max_row_id"], axis=1)

# verify all session_ids have exactly predict_rows rows
tmp = dat_long.filter(["id","session_id","row_id"])
tmp = tmp.sort_values(by = ["id","session_id","row_id"])
tmp = tmp.assign(Count=1).groupby(["id","session_id"])[["Count"]].count()
tmp = tmp.reset_index()

tmp["Count"].value_counts()
dat_long["id"] = dat_long["id"].astype(int)
dat["id"] = dat["id"].astype(int)
dat_long["row_id"] = dat_long["row_id"].astype(int)
dat["row_id"] = dat["row_id"].astype(int)

dat_long = pd.merge(left=dat_long, right=dat, on=["id","row_id"], how='left')
dat_long = pd.merge(left=dat_long, right=ids, on=["id"], how='left')

dat_long = dat_long.sort_values(by = ["id","session_id","row_id"])

# set future data to masked value
for col in dat_long.loc[:, "inicu":"vtother"].columns:
    dat_long.loc[dat_long["row_id"] > dat_long["session_id"], col] = -1
    dat_long.loc[dat_long["row_id"] < 1, col] = -1

# before icu admit
dat_long.loc[dat_long["row_id"] < 1, 'inicu'] = 0
dat_long = dat_long.sort_values(by = ["id","session_id","row_id"])

X_train = dat_long[(dat_long["train"] == 1)]
X_valid = dat_long[(dat_long["train"] == 0)]
X_ztest = dat_long[(dat_long["train"] == -999)]

ids_train = dat_long[(dat_long["train"] == 1)].groupby(["id","session_id"]).head(n=1).filter(items=["id","session_id","clust"])
ids_valid = dat_long[(dat_long["train"] == 0)].groupby(["id","session_id"]).head(n=1).filter(items=["id","session_id","clust"])
ids_ztest = dat_long[(dat_long["train"] == -999)].groupby(["id","session_id"]).head(n=1).filter(items=["id","session_id","clust"])

y_train = ids_train.filter(items=["clust"])
y_valid = ids_valid.filter(items=["clust"])
y_ztest = ids_ztest.filter(items=["clust"])

X_train.shape[0] / y_train.shape[0]

# keep one row for the neural network
X1_train = X_train.groupby(["id","session_id"]).tail(n=1)
X1_valid = X_valid.groupby(["id","session_id"]).tail(n=1)
X1_ztest = X_ztest.groupby(["id","session_id"]).tail(n=1)

X1_train = X1_train.loc[:, "age":"aids_hist"]
X1_valid = X1_valid.loc[:, "age":"aids_hist"]
X1_ztest = X1_ztest.loc[:, "age":"aids_hist"]

# subset to features
X_train = X_train.loc[:,"inicu":"vtother"]
X_valid = X_valid.loc[:,"inicu":"vtother"]
X_ztest = X_ztest.loc[:,"inicu":"vtother"]

print('LSTM training data shape:', X_train.shape)
print('X columns:', X_train.columns)

print('NN training data shape:', X1_train.shape)
print('X1 columns:', X1_train.columns)

# For sequential data features (X)
sequential_features = X_train.columns.tolist()  # Replace X_train with your original DataFrame
print("Sequential Features:", sequential_features)

# For summary data features (X1)
summary_features = X1_train.loc[:, "age":"aids_hist"].columns.tolist()    
print("Summary Features:", summary_features)

# set the proper shapes
X1_train = np.array(X1_train)
X1_valid = np.array(X1_valid)
X1_ztest = np.array(X1_ztest)

X1_train = np.reshape(X1_train, (y_train.shape[0], X1_train.shape[1]))
X1_valid = np.reshape(X1_valid, (y_valid.shape[0], X1_valid.shape[1]))
X1_ztest = np.reshape(X1_ztest, (y_ztest.shape[0], X1_ztest.shape[1]))

# set the proper shapes
X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_ztest = np.array(X_ztest)

X_train = np.reshape(X_train, (y_train.shape[0], predict_rows, X_train.shape[1]))
X_valid = np.reshape(X_valid, (y_valid.shape[0], predict_rows, X_valid.shape[1]))
X_ztest = np.reshape(X_ztest, (y_ztest.shape[0], predict_rows, X_ztest.shape[1]))

# Shuffle
X_train, X1_train, y_train, ids_train = shuffle(X_train, X1_train, y_train, ids_train, random_state=0)
X_valid, X1_valid, y_valid, ids_valid = shuffle(X_valid, X1_valid, y_valid, ids_valid, random_state=0)


y_train = y_train - 1
y_valid = y_valid - 1
y_ztest = y_ztest - 1

# Ensure the target labels are integers
y_train = y_train.astype("int32")
y_valid = y_valid.astype("int32")
y_ztest = y_ztest.astype("int32")

y_train = y_train.to_numpy()  # Convert DataFrame to NumPy array
y_valid = y_valid.to_numpy()
y_ztest = y_ztest.to_numpy()

y_train_time_steps = np.tile(y_train.reshape(-1, 1), (1, predict_rows))
y_valid_time_steps = np.tile(y_valid.reshape(-1, 1), (1, predict_rows))


# **Input for sequential data**
input1 = tf.keras.Input(shape=(predict_rows, X_train.shape[2]))
x1 = tf.keras.layers.Masking(mask_value=-1)(input1)

# **CNN Path**
x1 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, kernel_initializer=tf.keras.initializers.HeNormal(),
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation='relu', padding='same')(x1)
x1 = tf.keras.layers.BatchNormalization(momentum=0.9)(x1)  
x1 = tf.keras.layers.SpatialDropout1D(0.5)(x1)  

x1 = tf.keras.layers.Conv1D(filters=64, kernel_size=5           , kernel_initializer=tf.keras.initializers.HeNormal(),
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation='relu', padding='same')(x1)
x1 = tf.keras.layers.BatchNormalization(momentum=0.9)(x1)
x1 = tf.keras.layers.SpatialDropout1D(0.5)(x1)

# **GRU**
x1 = tf.keras.layers.GRU(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                          dropout=0.3, recurrent_dropout=0.2)(x1)

# **Input for summary data**
input2 = tf.keras.Input(shape=(X1_train.shape[1],))
x2 = tf.keras.layers.Dense(16, kernel_initializer=tf.keras.initializers.HeNormal(),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation='relu')(input2)
x2 = tf.keras.layers.BatchNormalization(momentum=0.9)(x2)

# **Merge paths**
merged = tf.keras.layers.concatenate([x1, x2])
merged = tf.keras.layers.Dense(16, activation='relu')(merged)
merged = tf.keras.layers.BatchNormalization(momentum=0.9)(merged)  
merged = tf.keras.layers.Dropout(0.5)(merged)  
final_output = tf.keras.layers.Dense(4)(merged)  # No softmax

# **Define Model**
model = tf.keras.Model(inputs=[input1, input2], outputs=final_output)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='.../optimized_cnn.keras',
    save_best_only=True,
    mode='min',
    monitor='val_loss'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Reduced patience for faster convergence
    start_from_epoch=5
)

tf.random.set_seed(42)

# **Set Optimizer with Selected Learning Rate**
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, clipnorm=1.0, clipvalue=1)

# **Compile Model**
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# **Learning Rate Scheduler**
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)


# **Model Summary**
model.summary()

# **Train Model**
history = model.fit(
    [X_train, X1_train],
    y_train,
    epochs=50,
    batch_size=256, 
    validation_data=([X_valid, X1_valid], y_valid),
    callbacks=[lr_scheduler, early_stopping],  
    verbose=2
)