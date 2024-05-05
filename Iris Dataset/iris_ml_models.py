from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC 
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

class IrisModels:
    def __init__(self,input,target,test_size, data):
        self.x, self.y = input,target
        self.data = data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input, target, test_size=test_size) 
        self.x_array, self.y_array = np.array(self.x), np.array(self.y)

    def linear_regression(self,fit_intercept = True, copyx = True,jobs=None,positive=False ,draw_reg_line = False):
        model = LinearRegression(fit_intercept=fit_intercept, copy_X=copyx,n_jobs=jobs,positive=positive)
        model.fit(self.x_train,self.y_train)
        pred = model.predict(self.x_test)
        print(f"Mean Squared Error of Linear Regression Model: {mse(self.y_test,pred)}")
        if draw_reg_line:
            sns.regplot(data=self.data, x="sepal length (cm)", y="sepal width (cm)", line_kws=dict(color='r'))
            plt.title("Regression Line")
            plt.xlabel("Sepal Length (cm)")
            plt.ylabel("Sepal Width (cm)")
        return mse(self.y_test,pred)
    
    def decision_tree(self, criterion = 'entropy', random_state=None, visualize=False):
        model = DecisionTreeClassifier(criterion=criterion, random_state=random_state)
        decision_tree_fitter = model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        print(f"Mean Squared Error of Decision Tree Model: {mse(self.y_test,pred)}")
        if visualize:
            plot_tree(decision_tree_fitter)
        return mse(self.y_test,pred)

    def support_vector_machines(self, max_iter=1000, random_state=None, visualize=False, C= 1, gamma=10, kernel : {'linear','poly','rbf','precomputed','sigmoid'}='rbf'):
        model = SVC(max_iter=max_iter, random_state=random_state,gamma=gamma,C=C, kernel=kernel)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        if visualize:
            pass
        return mse(self.y_test,pred)

    def logistic_regression(self,visualize=False,max_iter=1000,random_state = None, penalty : {'l2','l1','elasticnet'} ='l2'):
        model = LogisticRegression(random_state=random_state, max_iter=max_iter, penalty=penalty)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        if visualize:
            sns.regplot(data=self.data, x="sepal length (cm)", y="sepal width (cm)",logistic=True,ci=None, line_kws=dict(color='r'))
            plt.xlabel("Sepal Length (cm)")
            plt.ylabel("Sepal Width (cm)")
        return mse(self.y_test, pred)