import numpy as np

class Linear_regression():

  #Initiating the Parameters Learning_rate and No. of Iterations 
  def __init__(self, learning_rate, no_of_iterations):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations

  def fit(self, X, Y):

    # Number of Training Examples and Number of Features 
    self.m, self.n = X.shape # No. of Rows and Columns 

    #Initiating the Weight and Bias 
    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X 
    self.Y = Y

 # implementing Gradient Descent
    for i in range(self.no_of_iterations):
      self.update_weights()


  def update_weights(self):

    Y_prediction = self.predict(self.X)

    #Calculate Gradients Descent 
    dw = -(2* (self.X.T).dot(self.Y - Y_prediction)) / self.m

    db = - 2 * np.sum(self.Y - Y_prediction)/self.m
    
    #Updating the Weights 
    self.w = self.w - self.learning_rate*dw 

    self.b = self.b - self.learning_rate*db

  def predict(self, X):
    return X.dot(self.w) + self.b    
