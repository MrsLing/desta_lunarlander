# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:59:46 2022

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#设置一个随机种子,
#生成固定数值的随机整数数组
#seed=np.random.seed(42)
#data=np.random.randint(0,10,size=(10,10))

X=np.arange(0,10.1,1)
Y=np.arange(0,10.1,1)
X,Y=np.meshgrid(X,Y)
#d7=(np.sqrt(X**2+Y**2)+1)**2
#d8=(np.sqrt((X-0)**2+(Y-10)**2)+1)**2
#r=(30/d7+80/d8)/((np.sqrt((X-10)**2+(Y-10)**2)+1)**2)
#d1:energy1(5,4),d2:energy2(6,7),d3:energy3(8,7),d4:energy4(8,8)| d:sink(10,10)
d1=(np.sqrt((X-5)**2+(Y-5)**2)+1)**2
d2=(np.sqrt((X-4.5)**2+(Y-6)**2)+1)**2
d3=(np.sqrt((X-3)**2+(Y-3)**2)+1)**2
d4=(np.sqrt((X-5)**2+(Y-6)**2)+1)**2
d5=(np.sqrt((X-5)**2+(Y-4)**2)+1)**2
d6=(np.sqrt((X-4)**2+(Y-5)**2)+1)**2
d=(np.sqrt((X-5)**2+(Y-10)**2)+1)**2
r=(100/d1+80/d2+50/d3+2600/d4+100/d5+80/d6)/d

#print(data)
#fig=plt.figure(figsize=(10,8))
a = plt.gca()#获取边框
bwith = 2 #边框宽度设置为2
a.spines['top'].set_color('black')  # 设置上‘脊梁’为红色
a.spines['bottom'].set_color('black') 
a.spines['left'].set_color('black') 
a.spines['right'].set_color('black') 
a.spines['bottom'].set_linewidth(bwith)
a.spines['left'].set_linewidth(bwith)
a.spines['top'].set_linewidth(bwith)
a.spines['right'].set_linewidth(bwith)
plt.xticks(fontsize=16) #x轴刻度字体大小
plt.yticks(fontsize=16) #y轴刻度字体大小
#这就是所谓的第一种情况哦
h=plt.contourf(X,Y,r)
cb=plt.colorbar(h)
cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }
plt.savefig("./figures/best_point.pdf",format='pdf', dpi=1000, bbox_inches = 'tight')
#cb.set_label('colorbar',fontdict=font) #设置colorbar的标签字体及其大小

"""
能量分布图
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 
import xlrd
data = xlrd.open_workbook(r'C:/Users/dell/Desktop/figures/energy1.xlsx')
table = data.sheets()[1]
x=table.col_values(0)


y1=table.col_values(1)
plt.plot(x, y1,label=u"$l_{0}$=(5,0),δ=0.1,sink=(10,10)",linewidth=2)
y2=table.col_values(2)
plt.plot(x, y2,label=u"$l_{0}$=(5,5),δ=0.1,sink=(10,10)",linewidth=2)
y3=table.col_values(3)
plt.plot(x, y3,label=u"$l_{0}$=(10,0),δ=0.1,sink=(5,10)",linewidth=2)
y4=table.col_values(4)
plt.plot(x, y4,label=u"$l_{0}$=(10,5),δ=0.1,sink=(5,10)",linewidth=2)

y6=table.col_values(6)
plt.plot(x, y6,label=u"$l_{0}$=(5,5),δ=1.2,sink=(10,10)",linewidth=2)

y8=table.col_values(8)
plt.plot(x, y8,label=u"$l_{0}$=(5,0),δ=1.2,sink=(10,10)",linewidth=2)


#y9=table.col_values(9)
#plt.plot(x, y9,label=u"$l_{0}$=(10,5),δ=0.1,sink=(5,10)",linewidth=2)
#y10=table.col_values(10)
#plt.plot(x, y10,label=u"$l_{0}$=(1,0),δ=2,sink=(5,10)",linewidth=2)


#y11=table.col_values(11)
#plt.plot(x, y11,label=u"$l_{0}$=(5,5),δ=0.1,sink=(10,10)",linewidth=2)

#y12=table.col_values(12)
#plt.plot(x, y12,label=u"$l_{0}$=(1,5),δ=3,sink=(10,10)",linewidth=2)


plt.xlabel(u'Slot',size=18)
plt.ylabel(u"Battery Level",size=18)
plt.legend(frameon=True,edgecolor="black",fontsize='large')
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
plt.grid(True)
    #plt.figure(figsize=(8,6))
plt.savefig("./figures/remaining-battery-capacity1.pdf",format='pdf', dpi=1000, bbox_inches = 'tight')
plt.show()    
    
"""


