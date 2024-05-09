from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC 
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

    def support_vector_machines(self, max_iter=1000, random_state=0, visualize=False, C= 1.0, gamma=.10, kernel : {'linear','poly','rbf','precomputed','sigmoid'} = 'rbf'):
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


    def plot_decision_regions(self,X, y, classifier, test_idx=None, resolution=0.02):

        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl) 


    def knn(self,k=3, metric='minkowski', leaf_size=30, visualize=False):
        model = KNeighborsClassifier(n_neighbors=k, metric=metric, leaf_size=leaf_size)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)

        sc = StandardScaler()
        sc.fit(self.x_train)
        self.x_train_std = sc.transform(self.x_train)
        self.x_test_std = sc.transform(self.x_test)

        if visualize:
            pass
        return mse(self.y_test,pred)
    
    def random_forest(self, estimator=20, criterion="gini",show_confusion_matrix=False ):
        model = RandomForestClassifier(n_estimators=estimator, criterion=criterion)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        if show_confusion_matrix:
            plt.figure(figsize=(10,7))
            cm = confusion_matrix(self.y_test, pred)
            sns.heatmap(cm, annot=True)
            plt.xlabel("Predicted")
            plt.ylabel("Truth")
        return mse(self.y_test, pred)
