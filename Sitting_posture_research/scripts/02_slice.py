import numpy as np
import os

BASE_DIR = r'C:\Users\User\Desktop\Sitting_posture_research'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'custom')

good = np.load(os.path.join(DATA_DIR, 'good_posture_data.npy'))
bad = np.load(os.path.join(DATA_DIR, 'bad_posture_data.npy'))

def create_windows(data, label):
    X, y = [], []
    for i in range(len(data) - 60):
        X.append(data[i:i+60])
        y.append(label)
    return np.array(X), np.array(y)

X_g, y_g = create_windows(good, 0)
X_b, y_b = create_windows(bad, 1)
X = np.concatenate([X_g, X_b], axis=0)
y = np.concatenate([y_g, y_b], axis=0)

np.save(os.path.join(BASE_DIR, 'data', 'X_final.npy'), X)
np.save(os.path.join(BASE_DIR, 'data', 'y_final.npy'), y)
print(f"Final Data Shape: {X.shape}")