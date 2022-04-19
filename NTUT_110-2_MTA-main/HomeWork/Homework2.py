import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

x = open("iris_x.txt")
y = open("iris_y.txt")
with open("iris_x.txt", 'r') as f:
    data_x = []
    for i in f:
        index = i.split("\t")
        data = [float(index[0]), float(index[1]),
                float(index[2]), float(index[3])]
        data_x.append(data)
with open("iris_y.txt", 'r') as f:
    data_y = []
    for index in f:
        data_y.append(int(index.replace("\n", "")))
X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
    data_x, data_y, test_size=0.2, random_state=20220413)


x_train = np.array(X_train_sk)
y_train = np.array(y_train_sk)
x_test = np.array(X_test_sk)
y_test = np.array(y_test_sk)

linear = LinearRegression()
linear.fit(x_train, y_train)
y_predict = linear.predict(x_test)
MSE = np.mean((y_predict - y_test) ** 2)
plt.figure()


for i in range(3):
    pos = np.where(y_test == i)[0]
    plt.plot(y_predict[pos], y_test[pos], '*')
plt.plot([1.5, 1.5], [0, 2], 'k:')
plt.plot([0.5, 0.5], [0, 2], 'k:')
plt.title('MSE:{:.6f}'.format(MSE))
plt.show()

class QDA():
    def ___init__(self):
        self.mu = np.array([])
        self.cov = np.array([])

    def fit(self, data_train, label_train):
        mu, cov = [], []
        for i in range(np.max(label_train) + 1):
            pos = np.where(label_train == i)[0]
            tmp_data = data_train[pos, :]
            tmp_cov = np.cov(np.transpose(tmp_data))
            tmp_mu = np.mean(tmp_data, axis=0)
            mu.append(tmp_mu)
            cov.append(tmp_cov)
        self.mu = np.array(mu)
        self.cov = np.array(cov)

    def predict(self, x_test):
        d_value = []
        for tmp_mu, tmp_cov in zip(self.mu, self.cov):
            d = len(tmp_mu)
            zero_center_data = x_test - tmp_mu
            tmp = np.dot(zero_center_data.transpose(), np.linalg.inv(tmp_cov))
            tmp = -0.5 * np.dot(tmp, zero_center_data)
            tmp1 = (2 * np.pi) ** (-d / 2) * np.linalg.det(tmp_cov) ** (-0.5)
            tmp = tmp1 * np.exp(tmp)
            d_value.append(tmp)
        d_value = np.array(d_value)
        return np.argmax(d_value), d_value


sel = QDA()
sel.fit(x_train, y_train)
ans_pred= []
ans_set = []
for i in range(len(x_test)):
    pred , set = sel.predict(x_test[i])
    ans_pred.append(pred)
cm = confusion_matrix(y_test, ans_pred)
acc = np.diag(cm).sum() / cm.sum()
print('confusion_matrix (QDA):\n{}'.format(cm))
print('confusion_matrix (QDA,acc):{}'.format(acc))


qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda = qda.fit(x_train, y_train)
y_pred = qda.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = np.diag(cm).sum() / cm.sum()
print('confusion_matrix (QDA):\n{}'.format(cm))
print('confusion_matrix (QDA,acc):{}'.format(acc))
