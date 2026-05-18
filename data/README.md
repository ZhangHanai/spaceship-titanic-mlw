# Data directory

This repository does not commit Kaggle data files or generated preprocessing packages.
Download the data from the Kaggle Spaceship Titanic competition and place files here.

## SVM inputs

Place the raw Kaggle CSV files directly in this directory:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

`sample_submission.csv` is useful for reference, but the SVM script only requires `train.csv` and `test.csv`.

## LightGBM inputs

Place either the zip package:

```text
data/spaceship_catboost_preprocessed_package.zip
```

or an extracted directory:

```text
data/spaceship_catboost_preprocessed_package/
├── X_train_catboost_features.csv
├── X_test_catboost_features.csv
├── y_train_with_ids.csv
├── test_passenger_ids.csv
└── catboost_preprocessing_metadata.json
```

The LightGBM script will extract the zip automatically if the extracted folder is not already present.
