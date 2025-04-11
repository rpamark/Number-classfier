from sklearn.datasets import fetch_openml
import easygui
from PIL import Image
import numpy as np
print('Downloading mnist_784...')
mnist=fetch_openml('mnist_784',as_frame=False)#下载数据
print('Data downloaded successfully.')
x,y=mnist.data,mnist.target
#print(x.shape)
#print(y.shape)
import matplotlib.pyplot as plt
some_digit = x[0]
x_train,x_test,y_train,y_test=x[:60000],x[60000:],y[:60000],y[60000:]#x_train:用于训练，y_train:对应的答案，x_test:用于测试，y_test:对应的答案
from sklearn.svm import SVC
svm_clf=SVC(random_state=42)
svm_clf.fit(x_train[:2000],y_train[:2000])#只针对2000个数据训练
plt.imshow(some_digit.reshape(28, 28), cmap='binary')
plt.axis('off')
plt.show()
print('Result for some_digit:')
print(svm_clf.predict([some_digit]))
easygui.msgbox('Result for some_digit:'+str(svm_clf.predict([some_digit])))
some_digit_scores=svm_clf.decision_function([some_digit])
print(some_digit_scores.round(2))#输出决策分数
#计算precision
from sklearn.metrics import precision_score
import numpy as np
y_pred = svm_clf.predict(x_test)
precision = precision_score(y_test, y_pred, average='macro')
print('Precision on the entire test set:', precision)


