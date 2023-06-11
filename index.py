import pandas as pandas
import plot.index as plot
import numpy as np

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
        self.costs = np.zeros((201, 201))
        self.ws = np.arange(0, 0)
        self.bs = np.arange(0, 0)

        self.loseHistoryList = []
        self.wHistoryList = []
        self.bHistoryList = []

    ## function lose
    ## (真實數據 - 預測值 的 平方)
    ## lose = (y - y_predict)^2
    def loseFunction(self, w, b) -> float:
        y_predict = self.predictFunction(w, b)
        lose = (self.y - y_predict)**2
        return lose.sum() / len(self.x)

        # plt.interact((-100, 10 0, 1), (100, 100, 1))
        # self.plt.open()

    # 梯度下降 寒士
    # w方向微分 後 = 2*x*(w*x + b) -y
    # b方向微分 後 = 2*(w*x + b) -y
    def gradientFunction(self, w, b):
        w_gradient = (2 * self.x * (w * self.x + b) - self.y).mean()
        b_gradient = (2 * (w * self.x + b) - self.y).mean()
        # print(w_gradient, b_gradient)
        return w_gradient, b_gradient
    
    # 計算 lose
    # wRange = [start, end]
    # bRange = [start, end]
    def computed_lose(self, wRange, bRange):
        self.ws = np.arange(wRange[0], wRange[1])
        self.bs = np.arange(bRange[0], bRange[1])

        i = 0
        for w in self.ws:
            j = 0
            for b in self.bs:
                returnCost = self.loseFunction(w, b)
                self.costs[i, j] = returnCost
                j += 1
            i += 1

    # 計算 梯度下降
    def computed_gradient(self, wInit = 0, bInit = 0, iteration = 1000, learning_rate = 0.001):
        w = wInit
        b = bInit
        for i in range(iteration):
            w_gradient, b_gradient = self.gradientFunction(w, b)

            # 經過 梯度下降後 更新後的 w b
            w = w - w_gradient * learning_rate
            b = b - b_gradient * learning_rate

            lose = self.loseFunction(w, b)

            self.loseHistoryList.append(lose)
            self.wHistoryList.append(w)
            self.bHistoryList.append(b)

            if i % 1000 == 0:
                print(f"iteration {i}  w: {w: .2e} b: {b: .2e} cost: {lose: .2e}")

    ## 繪圖 出 lose function
    def drawComputedLost(self, wRange):
        ax = self.plt.axes(projection = "3d")
        ax.xaxis.set_pane_color((0,0,0))
        ax.yaxis.set_pane_color((0,0,0))
        ax.zaxis.set_pane_color((0,0,0))
        ax.view_init(45, -120)

        ax.set_title('w b 對應的 costs')
        ax.set_xlabel('w')
        ax.set_ylabel('b')
        ax.set_ylabel('cost')


        b_grid, w_grid = np.meshgrid(self.bs, self.ws)
        ax.plot_surface(b_grid, w_grid, self.costs)

        self.plt.show()
    
    def drawComputedGredientDecent():
        print()

class DP:
    def __init__(self, x, y) -> None:
        self.lost = lose(x, y)

if __name__ == '__main__':
    data = pandas.read_csv(url)
    x = data['YearsExperience']
    y = data['Salary']
    w = 0
    b = 0
    dp = DP(x, y)
    dp.lost.computed_gradient(w, b, 10000, 0.001)