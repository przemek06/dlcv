import pandas as pd
import numpy as np
from pathlib import Path

metadata_path="animal-clef-2025/metadata.csv"
output_dir="animal-clef-2025/splits"
unseen_s_valid_ratio=0.5
unseen_m_individuals=80
seen_valid_ratio=0.1
seen_test_ratio=0.1
seed=42

def split_clef_dataset():
    np.random.seed(seed)

    df = pd.read_csv(metadata_path)
    db = df[df['split'] == 'database'].copy()

    identity_counts = db['identity'].value_counts()
    s_identities = set(identity_counts[identity_counts == 1].index)
    m_identities = set(identity_counts[identity_counts > 1].index)

    s_list = list(s_identities)
    m_list = list(m_identities)
    np.random.shuffle(s_list)
    np.random.shuffle(m_list)

    n_s_valid = int(len(s_list) * unseen_s_valid_ratio)
    s_valid_unseen_ids = set(s_list[:n_s_valid])
    s_test_unseen_ids = set(s_list[n_s_valid:])

    n_m_unseen_valid = unseen_m_individuals // 2
    n_m_unseen_test = unseen_m_individuals - n_m_unseen_valid
    m_valid_unseen_ids = set(m_list[:n_m_unseen_valid])
    m_test_unseen_ids = set(m_list[n_m_unseen_valid:n_m_unseen_valid + n_m_unseen_test])
    m_seen_ids = set(m_list[n_m_unseen_valid + n_m_unseen_test:])

    valid_unseen_images = []
    test_unseen_images = []

    for idx, row in db.iterrows():
        identity = row['identity']
        if identity in s_valid_unseen_ids:
            valid_unseen_images.append(idx)
        elif identity in s_test_unseen_ids:
            test_unseen_images.append(idx)
        elif identity in m_valid_unseen_ids:
            valid_unseen_images.append(idx)
        elif identity in m_test_unseen_ids:
            test_unseen_images.append(idx)

    train_images = []
    valid_seen_images = []
    test_seen_images = []

    for identity in m_seen_ids:
        id_images = db[db['identity'] == identity].index.tolist()
        np.random.shuffle(id_images)

        n_images = len(id_images)
        n_valid = max(1, int(n_images * seen_valid_ratio))
        n_test = max(1, int(n_images * seen_test_ratio))
        n_train = n_images - n_valid - n_test

        if n_train < 1:
            n_train = 1
            n_valid = (n_images - 1) // 2
            n_test = n_images - 1 - n_valid

        train_images.extend(id_images[:n_train])
        valid_seen_images.extend(id_images[n_train:n_train + n_valid])
        test_seen_images.extend(id_images[n_train + n_valid:])

    train_df = db.loc[train_images].copy()
    valid_seen_df = db.loc[valid_seen_images].copy()
    valid_unseen_df = db.loc[valid_unseen_images].copy()
    test_seen_df = db.loc[test_seen_images].copy()
    test_unseen_df = db.loc[test_unseen_images].copy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    valid_seen_df.to_csv(output_path / "valid_seen.csv", index=False)
    valid_unseen_df.to_csv(output_path / "valid_unseen.csv", index=False)
    test_seen_df.to_csv(output_path / "test_seen.csv", index=False)
    test_unseen_df.to_csv(output_path / "test_unseen.csv", index=False)

    return train_df, valid_seen_df, valid_unseen_df, test_seen_df, test_unseen_df


if __name__ == "__main__":
    split_clef_dataset()
