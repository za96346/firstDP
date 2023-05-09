import pandas as pandas
import plot.index as plot

url = './Salary_Data.csv'


data = pandas.read_csv(url)
# print(data)
x = data['YearsExperience']
y = data['Salary']

drawer = plot.drawer(x, y)

## function 
## y = w*x + b
w = 0
b = 10

# 預測值
y_predict = x*w + b

# drawer.addLine(x, y_predict)

drawer.interact((-100, 100, 1), (100, 100, 1))
drawer.open()