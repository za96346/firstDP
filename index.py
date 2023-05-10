import pandas as pandas
import plot.index as plot

url = './Salary_Data.csv'

class linear(plot.drawer):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y

    ## function 
    ## y_predict = w*x + b
    def predictFunction(self, w, b):
        # 預測值
        y_predict = self.x * w + b
        # self.plt.addLine(self.x, y_predict)
        return y_predict
    
        ## 繪圖 出 真實數據
    def draw(self, **kwargs):
        self.plt.scatter(
            self.x,
            self.y,
            marker="h",
            color="green",
            label="真實數據"
        )
        self.plt.xlabel('年資')
        self.plt.ylabel('月薪')

        # x 最大最小值
        if 'xRange' in kwargs:
            self.plt.xlim(kwargs.xRange)
        else:
            self.plt.xlim([0, 12])

        # y 最大最小值
        if 'yRange' in kwargs:
            self.plt.ylim(kwargs.yRange)
        else:
            self.plt.ylim([-60, 140])


# 損失 function
class lose(linear):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)
        self.costs = []

    ## function lose
    ## (真實數據 - 預測值 的 平方)
    ## lose = (y - y_predict)^2
    def loseFunction(self, w, b) -> float:
        y_predict = self.predictFunction(w, b)
        lose = (self.y - y_predict)**2
        return lose.sum() / len(self.x)

        # plt.interact((-100, 10 0, 1), (100, 100, 1))
        # self.plt.open()
    
    # 計算 lose
    # wRange = [start, end]
    # bRange = [start, end]
    def computed(self, wRange, bRange):
        for w in range(wRange[0], wRange[1]):
            returnCost = self.loseFunction(w, 0)
            self.costs.append(returnCost)

    ## 繪圖 出 lose function
    def draw(self, wRange):
        self.plt.scatter(range(wRange[0], wRange[1]), self.costs)
        self.plt.ylabel = 'b'
        self.plt.xlabel = 'w'
        self.plt.show()

class DP:
    def __init__(self, x, y) -> None:
        self.lost = lose(x, y)

if __name__ == '__main__':
    data = pandas.read_csv(url)
    x = data['YearsExperience']
    y = data['Salary']
    dp = DP(x, y)
    dp.lost.computed([-100, 101], 0)
    dp.lost.draw([-100, 101])