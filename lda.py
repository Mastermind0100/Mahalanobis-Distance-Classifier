import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from alive_progress import alive_it

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

testing_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

image_arr = []
labels_arr = []

print("Customizing Dataset... ")
bar = alive_it(training_data)
for data in bar:
    img, label_id = data
    image_arr.append(img.numpy().ravel())
    labels_arr.append(label_id)
print("Done! \u2713\n")

clf = LinearDiscriminantAnalysis()
clf.fit(image_arr, labels_arr)

print("Testing... ")
testing_img_arr = []
testing_labels_arr = []
for data in bar:
    img, label_id = data
    testing_img_arr.append(img.numpy().ravel())
    testing_labels_arr.append(label_id)

outputs = []
bar = alive_it(list(range(len(testing_img_arr))))
for i in bar:
    test_img = testing_img_arr[i]
    pred = clf.predict([test_img])[0]
    if pred == testing_labels_arr[i]:
        outputs.append(1)
    else:
        outputs.append(0)
print("Done! \u2713\n")

print(sum(outputs)/len(outputs))

# print(clf.predict([image_arr[0]]))
# print(labels_arr[0])