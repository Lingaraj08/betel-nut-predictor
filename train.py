# train.py
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

data_dir = "data/betelnut_images/dataset"
X, y = [], []
class_names = os.listdir(data_dir)

for label in class_names:
    folder = os.path.join(data_dir, label)
    for img_file in os.listdir(folder):
        path = os.path.join(folder, img_file)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        X.append(img.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/classifier.pkl")
