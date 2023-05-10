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
    def __init__(self):
        self.plt = plt

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