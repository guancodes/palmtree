from palmtree.model import load_model
from palmtree.data import load_data
from sklearn.metrics.classification import accuracy_score

print("loading data ...")
data = load_data(False)

print("loading model ...")
mod = load_model("wicked.mod")

print("predicting ...")
y_truth = []
y_predict = []
for img, truth, filename in zip(*data):
    prediction = mod.predict([img])
    if truth != prediction:
        print("%d != %d for %s" % (truth, prediction, filename))
    y_truth.append(truth)
    y_predict.append(prediction)

print("accuracy:", accuracy_score(y_truth, y_predict))
print("done.")
