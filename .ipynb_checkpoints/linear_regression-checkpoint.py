import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

class line:
    def __init__(self,slope=1,intercept=0,variance=1,points=10):
        """Instantiates a line with a given slope, intercept, variance, and number of points"""
        self.slope = slope
        self.intercept = intercept
        self.variance = variance
        self.points = points
        self.x = self.__return_x(self.intercept,self.variance,self.slope,self.points)
        self.y = self.__return_y(self.x,self.intercept,self.variance,self.slope,self.points)
        
    def __return_x(self,intercept,variance,slope,points):
        x = np.arange(start=0,stop=points)
        return x
    
    def __return_y(self,x,intercept,variance,slope,points):
        def __residuals(variance=variance):
            return [np.random.normal(scale=variance**2,loc=0) for i in range(points)] 
        def f(x,intercept,slope):
            return slope*x+intercept
        y = f(x,intercept,slope)
        y_with_variance = y + __residuals()
        return y_with_variance
    
    def plot(self):
        return plt.scatter(self.x,self.y)     
    
class slr:
    def __init__(self,line):
        self.x = line.x
        self.y = line.y
        self.beta1_hat = None
        self.beta0_hat = None
    
    def fit(self):
        '''Fits data via OLS simple linear regression'''
        numerator = 0
        denominator = 0
        x_bar = sum(self.x)/len(self.x)
        y_bar = sum(self.y)/len(self.y)
        n = len(self.x)
        for i in range(n):
            xi = self.x[i]
            yi = self.y[i]
            numerator+=(xi*yi)-(n*x_bar*y_bar)
            denominator+=(xi**2)-n*(x_bar**2)

        self.beta1_hat=numerator/denominator
        self.beta0_hat = y_bar-beta1_hat*x_bar      
    
    def predict(self,x):
        return self.beta0_hat+self.beta1_hat*x
    
    def plot(self):
        fig, ax = plt.subplots()
        ax.scatter(self.x, self.y, c='blue')
        line = mlines.Line2D(xdata = self.x, ydata = self.predict(np.array(self.x)), color='red')
        ax.add_line(line)
        return plt  
