import groot.datasets as datasets_module

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np

datasets = ["banknote-authentication", "blood-transfusion", "breast-cancer", "cylinder-bands", "diabetes", "haberman", "ionosphere", "wine"]
data_dir = "data/"

for dataset in datasets:
    # From the groot.datasets module, find the function with name load_<selected_dataset> then execute it
    X, y = getattr(datasets_module, f"load_{dataset.replace('-', '_')}")()[1:3]
    scaler = MinMaxScaler()

    # Perform a 80%/20% train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)
    X_train = scaler.fit_transform(X_train)

    # Make sure not to cheat on the test set when scaling
    X_test = np.clip(scaler.transform(X_test), 0.0, 1.0)

    np.save(data_dir + f"X_train_{dataset}.npy", X_train, allow_pickle=False)
    np.save(data_dir + f"X_test_{dataset}.npy", X_test, allow_pickle=False)

    np.save(data_dir + f"y_train_{dataset}.npy", y_train, allow_pickle=False)
    np.save(data_dir + f"y_test_{dataset}.npy", y_test, allow_pickle=False)
