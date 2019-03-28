import numpy as np

training_set = np.loadtxt("mnist_train.csv", delimiter=',', dtype=int)
testing_set = np.loadtxt("mnist_test.csv", delimiter=',', dtype=int)
test_labels = testing_set[:,0]

print("loaded data.")

# resize images
from scipy.misc import imresize
def resize(a):
    return imresize(a.reshape((28,28)), (17,17)).flatten()
shrunk_training_set = np.apply_along_axis(resize, 1, training_set[:,1:])
shrunk_testing_set = np.apply_along_axis(resize, 1, testing_set[:,1:])
print("resized images.")

# apply thresholding
from skimage.filters import threshold_otsu
def otsu_thresh(a):
    thresh = threshold_otsu(a)
    return a > thresh

threshed_training_set = np.apply_along_axis(otsu_thresh, 1, shrunk_training_set)
threshed_testing_set = np.apply_along_axis(otsu_thresh, 1, shrunk_testing_set)
print("applied thresholding.")


# add labels as categoricals
training_set = np.append(
                    threshed_training_set,
                    training_set[:,0].reshape((training_set.shape[0],1)),
                    axis=1
                )
testing_set = np.append(
                    threshed_testing_set,
                    np.tile(np.array([np.nan]),(testing_set.shape[0],1)),
                    axis=1
                )
print(testing_set.shape)
print("added labels.")

#train spn
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Context
print(training_set.shape)
spn_classification = learn_classifier(training_set,
                       Context(
                            parametric_types=[Categorical]*289+[Categorical]
                        ).add_domains(training_set),
                       learn_parametric, 289)

print("trained spn.")

import pickle
outfile = open('mnist_spn_trained','wb')
pickle.dump(spn_classification,outfile)
outfile.close()
print("pickled spn.")

#plot spn
from spn.io.Graphics import plot_spn
plot_spn(spn_classification, 'mnist_spn.png')


#test spn
from spn.algorithms.MPE import mpe
predictions = mpe(spn_classification, testing_set)

from sklearn.metrics import accuracy_score
print("accuracy:")
print(accuracy_score(test_labels, predictions[:,-1:]))
print("actuals:")
print(test_labels[:10])
print("predictions:")
print(predictions[:10,-1:])
