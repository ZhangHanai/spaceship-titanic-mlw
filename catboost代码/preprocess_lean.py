"""
Lean preprocessing for Spaceship Titanic.

Design goals (drives every decision below):
  1. ~20-30 features, no log/share/used redundancy stack
  2. Hard logical constraints between CryoSleep and Spend
  3. Group-aware imputation for HomePlanet, Destination, Side, Deck
  4. CatBoost-native categorical handling: keep strings, do NOT one-hot

Path convention: env var SPACETITANIC_DATA, fallback to CWD. Windows-safe.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get('SPACETITANIC_DATA', '.'))
SPEND_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def _group_mode_fill(df, col, by):
    """Fill missing in `col` with group-wise mode over `by`. Robust to all-NaN groups."""
    def first_mode(s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else np.nan
    fill = df.groupby(by, dropna=False)[col].transform(first_mode)
    return df[col].fillna(fill)


def _group_median_fill(df, col, by):
    fill = df.groupby(by, dropna=False)[col].transform('median')
    return df[col].fillna(fill)


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Build a lean feature set. Concatenate train+test for stable group/surname stats."""
    y = train['Transported'].astype(int).values
    n_train = len(train)

    df = pd.concat(
        [train.drop(columns=['Transported']), test],
        axis=0,
        ignore_index=True
    )

    # ---- 1. PassengerId split -------------------------------------------
    pid = df['PassengerId'].str.split('_', expand=True)
    df['Group'] = pid[0].astype(int)
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

    # ---- 2. Cabin split -------------------------------------------------
    cab = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cab[0]
    df['CabinNum'] = pd.to_numeric(cab[1], errors='coerce')
    df['Side'] = cab[2]

    # ---- 3. Surname (used only for imputation, drop before training) ----
    df['Surname'] = df['Name'].astype(str).str.split().str[-1]
    df.loc[df['Name'].isna(), 'Surname'] = np.nan

    # ---- 4. Logical Cryo <-> Spend constraints --------------------------
    # Step 4a: CryoSleep=True known + missing spend  -> 0
    for c in SPEND_COLS:
        df.loc[(df['CryoSleep'] == True) & (df[c].isna()), c] = 0.0

    # Step 4b: known spend > 0 -> force CryoSleep=False (overrides missing)
    known_spend = df[SPEND_COLS].fillna(0).sum(axis=1)
    spend_positive_known = (known_spend > 0)
    df.loc[spend_positive_known & df['CryoSleep'].isna(), 'CryoSleep'] = False

    # Step 4c: all 5 spend known and == 0 + Cryo missing -> True (very strong signal)
    all_known = df[SPEND_COLS].notna().all(axis=1)
    df.loc[
        df['CryoSleep'].isna() & all_known & (known_spend == 0),
        'CryoSleep'
    ] = True

    # ---- 5. Group-aware fills for HomePlanet / Destination / Side / Deck
    df['HomePlanet'] = _group_mode_fill(df, 'HomePlanet', 'Group')
    df['HomePlanet'] = _group_mode_fill(df, 'HomePlanet', 'Surname')
    df['HomePlanet'] = _group_mode_fill(df, 'HomePlanet', 'Deck')
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode().iloc[0])

    df['Destination'] = _group_mode_fill(df, 'Destination', 'Group')
    df['Destination'] = _group_mode_fill(df, 'Destination', 'Surname')
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode().iloc[0])

    df['Side'] = _group_mode_fill(df, 'Side', 'Group')
    df['Side'] = df['Side'].fillna(df['Side'].mode().iloc[0])

    df['Deck'] = _group_mode_fill(df, 'Deck', 'Group')
    df['Deck'] = _group_mode_fill(df, 'Deck', ['HomePlanet', 'Side'])
    df['Deck'] = df['Deck'].fillna(df['Deck'].mode().iloc[0])

    # CabinNum: median by Deck+Side then Deck
    df['CabinNum'] = _group_median_fill(df, 'CabinNum', ['Deck', 'Side'])
    df['CabinNum'] = _group_median_fill(df, 'CabinNum', 'Deck')
    df['CabinNum'] = df['CabinNum'].fillna(df['CabinNum'].median())

    # ---- 6. Remaining spend imputation (still-missing rows) -------------
    # By this point: rows with missing spend either had Cryo=False or unresolvable Cryo.
    # Use conditional medians, NOT blind 0.
    for c in SPEND_COLS:
        df[c] = _group_median_fill(df, c, ['HomePlanet', 'CryoSleep'])
        df[c] = _group_median_fill(df, c, 'CryoSleep')
        df[c] = df[c].fillna(df[c].median())

    # ---- 7. CryoSleep / VIP residual --------------------------------------
    # Still-missing CryoSleep at this point is genuinely ambiguous.
    # Cast to string with explicit 'Unknown' so CatBoost can learn it as own category.
    df['CryoSleep'] = df['CryoSleep'].astype('object').where(~df['CryoSleep'].isna(), 'Unknown')
    df['CryoSleep'] = df['CryoSleep'].astype(str)

    df['VIP'] = df['VIP'].fillna(False).astype(str)

    # ---- 8. Age ---------------------------------------------------------
    df['Age'] = _group_median_fill(df, 'Age', ['HomePlanet', 'CryoSleep'])
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['IsChild'] = (df['Age'] < 13).astype(int)

    # ---- 9. Spending aggregates (only the few that matter) --------------
    df['TotalSpend'] = df[SPEND_COLS].sum(axis=1)
    df['NoSpend'] = (df['TotalSpend'] == 0).astype(int)
    # log1p on TotalSpend ONLY (single representative log feature, not 6 of them)
    df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
    # Luxury vs Leisure split: known from EDA that these two groups behave differently
    df['LuxurySpend'] = df['RoomService'] + df['Spa'] + df['VRDeck']
    df['LeisureSpend'] = df['FoodCourt'] + df['ShoppingMall']

    # ---- 10. Group-aggregate features (only kept the ones with intuition)
    df['GroupTotalSpend'] = df.groupby('Group')['TotalSpend'].transform('sum')
    df['GroupChildrenCount'] = df.groupby('Group')['IsChild'].transform('sum')

    # ---- 11. Final feature list ----------------------------------------
    feature_cols = [
        # raw-ish
        'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
        'Age', 'IsChild',
        # cabin
        'Deck', 'CabinNum', 'Side',
        # group
        'Group', 'GroupSize', 'IsAlone',
        # spending
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'TotalSpend', 'NoSpend', 'LogTotalSpend',
        'LuxurySpend', 'LeisureSpend',
        # group aggregates
        'GroupTotalSpend', 'GroupChildrenCount',
    ]
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

    # Ensure cat cols are strings, no NaN
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna('Missing').replace({'nan': 'Missing'})

    X_full = df[feature_cols].copy()
    X_train = X_full.iloc[:n_train].reset_index(drop=True)
    X_test = X_full.iloc[n_train:].reset_index(drop=True)

    ids_train = df['PassengerId'].iloc[:n_train].reset_index(drop=True)
    ids_test = df['PassengerId'].iloc[n_train:].reset_index(drop=True)

    return X_train, X_test, y, ids_train, ids_test, feature_cols, cat_cols


def main():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    X_train, X_test, y, id_tr, id_te, feats, cats = build_features(train, test)
    print('X_train:', X_train.shape, '| X_test:', X_test.shape)
    print('Features (', len(feats), '):', feats)
    print('Categorical:', cats)
    print('Any NaN in X_train?', X_train.isna().any().any(),
          ' X_test?', X_test.isna().any().any())
    print('Target balance:', np.mean(y))

    X_train.to_csv(DATA_DIR / 'X_train_lean.csv', index=False)
    X_test.to_csv(DATA_DIR / 'X_test_lean.csv', index=False)
    pd.DataFrame({'PassengerId': id_tr, 'Transported': y.astype(bool)}).to_csv(
        DATA_DIR / 'y_train_lean.csv', index=False
    )
    pd.DataFrame({'PassengerId': id_te}).to_csv(
        DATA_DIR / 'test_ids_lean.csv', index=False
    )
    with open(DATA_DIR / 'lean_meta.json', 'w') as f:
        json.dump({'features': feats, 'categorical': cats}, f, indent=2)
    print('Saved lean preprocessing.')


if __name__ == '__main__':
    main()
