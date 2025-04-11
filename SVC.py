from sklearn.datasets import fetch_openml
import easygui
from PIL import Image
import numpy as np
print('Downloading mnist_784...')
mnist=fetch_openml('mnist_784',as_frame=False)#��������
print('Data downloaded successfully.')
x,y=mnist.data,mnist.target
#print(x.shape)
#print(y.shape)
import matplotlib.pyplot as plt
some_digit = x[0]
x_train,x_test,y_train,y_test=x[:60000],x[60000:],y[:60000],y[60000:]#x_train:����ѵ����y_train:��Ӧ�Ĵ𰸣�x_test:���ڲ��ԣ�y_test:��Ӧ�Ĵ�
from sklearn.svm import SVC
svm_clf=SVC(random_state=42)
svm_clf.fit(x_train[:2000],y_train[:2000])#ֻ���2000������ѵ��
plt.imshow(some_digit.reshape(28, 28), cmap='binary')
plt.axis('off')
plt.show()
print('Result for some_digit:')
print(svm_clf.predict([some_digit]))
easygui.msgbox('Result for some_digit:'+str(svm_clf.predict([some_digit])))
some_digit_scores=svm_clf.decision_function([some_digit])
print(some_digit_scores.round(2))#������߷���
#����precision
from sklearn.metrics import precision_score
import numpy as np
y_pred = svm_clf.predict(x_test)
precision = precision_score(y_test, y_pred, average='macro')
print('Precision on the entire test set:', precision)


