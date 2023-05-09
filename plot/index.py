import matplotlib.pyplot as plt
import wget
import matplotlib as mpl 
from matplotlib.font_manager import fontManager
from ipywidgets import interact

# wget.download("https://github.com/GrandmaCan/ML/raw/main/Resgression/ChineseFont.ttf")

# 加入 字型
fontManager.addfont("./plot/ChineseFont.ttf")
mpl.rc('font', family="ChineseFont")

class drawer:
    # 畫圖
    def __init__(self, x, y, **kwargs):
        self.plt = plt
        self.plt.scatter(
            x,
            y,
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
        

    # 打開
    def open(self):
        self.plt.legend()
        self.plt.show()
    
    # 新增線段
    def addLine(self, x, y):
        self.plt.plot(x, y, color="blue", label="預測線")
    
    # 互動元件 目前 失敗
    def interact(self, xRange, yRange):
        interact(self.addLine, x = xRange, y = yRange)

if __name__ == '__main__':
    drawer([], [])