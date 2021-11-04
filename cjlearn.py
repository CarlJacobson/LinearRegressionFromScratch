import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

class random_line_maker:
    def __init__(self,slope=1,intercept=0,variance=1,points=10):
        """Instantiates a line with a given slope, intercept, variance, and number of points"""
        self.slope = slope
        self.intercept = intercept
        self.variance = variance
        self.points = points
        self.x = self.__return_x(self.intercept,self.variance,self.slope,self.points)
        self.y = self.__return_y(self.x,self.variance,self.points)
        
    def __return_x(self,intercept,variance,slope,points):
        x = np.arange(start=0,stop=points,dtype=np.float64)
        return x
    
    def __return_y(self,x,variance,points):
        def __residuals(variance=variance):
            return [np.random.normal(scale=variance**2,loc=0) for i in range(points)] 
        def f():
            return self.slope*self.x+self.intercept
        y = f()
        y_with_variance = y + __residuals()
        return y_with_variance
    
    def plot(self):
        return plt.scatter(self.x,self.y)     
    
class models:
    class slr:
        def __init__(self):
            self.x = None
            self.y = None
            self.beta1_hat = None
            self.beta0_hat = None
            
        def formula(self,decimals = 3):
            return f"Y = {round(self.beta0_hat,decimals)} + {round(self.beta1_hat,decimals)} * x"

        def fit(self, line : random_line_maker = None, x = None,y = None):
            '''Fits data via OLS simple linear regression'''

            if (line == None) and (x == None or y == None):
                raise Exception('Input either array like input for x and y or line object as arguments')
            elif (line != None) and (x != None or y != None):
                print('elif')
                self.x = line.x
                self.y = line.y
            else:
                self.x = line.x
                self.y = line.y

            numerator = 0
            denominator = 0
            x_bar = sum(self.x)/len(self.x)
            y_bar = sum(self.y)/len(self.y)
            n = len(self.x)
            for i in range(n):
                xi = self.x[i]
                yi = self.y[i]
                numerator+=((yi-y_bar)*(xi-x_bar))
                denominator+=((xi-x_bar)**2)
            
            self.beta1_hat=numerator/denominator
            self.beta0_hat = y_bar-(self.beta1_hat*x_bar)    
            print(self.formula(decimals = 4))

        def predict(self,x):
            return self.beta0_hat+self.beta1_hat*x

        def plot(self):
            fig, ax = plt.subplots()
            ax.scatter(self.x, self.y, c='blue')
            line = mlines.Line2D(xdata = self.x, ydata = self.predict(np.array(self.x)), color='red')
            ax.add_line(line)
            return plt  
        

