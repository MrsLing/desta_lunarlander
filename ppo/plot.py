# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:23:57 2022

@author: dell
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 
import xlrd

#sink在(5,10)分别从（10,0）、（10,5）出发的rewards图
data = xlrd.open_workbook(r'E:/Phd/spyder/ppo/tra_evel1.xlsx')
table = data.sheets()[0]
x=table.col_values(0)#读取列的值
y=table.col_values(1)

plt.plot(x, y, 'r-',label=u"$l_{0}$=(5,5),δ=0.1",linewidth=2)


#plt.legend()
plt.xlabel(u'x',size=25)
plt.ylabel(u"y",size=25)
bwith = 2 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['top'].set_color('black')  # 设置上‘脊梁’为红色
ax.spines['bottom'].set_color('black') 
ax.spines['left'].set_color('black') 
ax.spines['right'].set_color('black') 
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.xticks(fontsize=15) #x轴刻度字体大小
plt.yticks(fontsize=15) #y轴刻度字体大小
plt.show()
plt.legend(frameon=True,edgecolor="black",fontsize='large')

