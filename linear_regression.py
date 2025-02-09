import numpy as np


class LinearRegression:
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.w=None
        self.b=None 

    def fit(self,X,y):
        m,n=X.shape
        self.w=np.zeros(n)
        self.b=0

        for i in range(self.epochs):
            y_pred = np.dot(X,self.w)+self.b
            error=y_pred-y

            dw = (2/m) * np.dot(X.T, error)
            db = (2/m) * np.sum(error)

            self.w-=self.learning_rate*dw
            self.b-=self.learning_rate*db

    
    def predict(self,X):
        return np.dot(X, self.w) + self.b  # y = Xw + b



if __name__ == "__main__":
    X=np.array([[1],[2],[3],[4],[5]])
    y=np.array([2,4,6,8,10])
    print(X.shape)
    

    model=LinearRegression(learning_rate=0.01,epochs=1000)
    model.fit(X,y)
    
    predictions = model.predict(X)
    
    
    print("Real Values:", y)
    print("Predicted Values:", predictions)
    