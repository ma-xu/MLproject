import numpy as np
from sklearn import svm


data= np.random.standard_normal([100,2])
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
model=clf.fit(data)
print(model)