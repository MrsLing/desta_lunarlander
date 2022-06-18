# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:28:26 2022

@author: dell
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 
# from pyecharts.charts import Line
#from pyecharts.charts import Line, Pie, Grid
#from pyecharts_snapshot.main import make_a_snapshot
import xlrd
"""
def rewards(file_name,lable1,lable2):
    data = xlrd.open_workbook(file_name)
    table = data.sheets()[0]
    x=table.col_values(0)
    y1=table.col_values(1)
    y11=table.col_values(2)
    y2=table.col_values(4)
    y22=table.col_values(5)
    plt.plot(x, y1, 'b-',label=lable1,linewidth=2,alpha=0.4)
    plt.plot(x, y11, 'b-',linewidth=2)
    plt.plot(x, y2, 'r-',label=lable2,linewidth=2,alpha=0.4)
    plt.plot(x, y22, 'r-',linewidth=2)
    plt.xlabel(u'Episode',size=18)
    plt.ylabel(u"Total Rewards",size=18)
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
    plt.savefig("./figures/rewards1_3.pdf",format='pdf', dpi=1000, bbox_inches = 'tight')
    plt.show()


def actorloss(file_name,lable1,lable2):
    data = xlrd.open_workbook(file_name)
    table = data.sheets()[1]
    x=table.col_values(0)
    y1=table.col_values(1)
    y2=table.col_values(4)
    plt.plot(x, y1, 'b-',label=lable1,linewidth=2)
    plt.plot(x, y2, 'r-',label=lable2,linewidth=2)
    plt.xlabel(u'Per 100 updates',size=18)
    plt.ylabel(u"Actor-Loss",size=18)
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
    plt.savefig("./figures/actor_loss1_3.pdf",format='pdf', dpi=1000, bbox_inches = 'tight')
    plt.show()
    
def criticloss(file_name,lable3,lable4):
    data = xlrd.open_workbook(file_name)
    table = data.sheets()[1]
    x=table.col_values(0)
    y11=table.col_values(2)
    y22=table.col_values(5)
    plt.plot(x, y11, 'b-',label=lable3,linewidth=2)
    plt.plot(x, y22, 'r-',label=lable4,linewidth=2)
    plt.xlabel(u'Per 100 updates',size=18)
    plt.ylabel(u"Critic-Loss",size=18)
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
    plt.savefig("./figures/critic_loss1_3.pdf",format='pdf', dpi=1000, bbox_inches = 'tight')
    plt.show()

"""

"""

def trajectory(file_name,lable1,lable2):
    data = xlrd.open_workbook(file_name)
    table = data.sheets()[2]
    x1=table.col_values(0)
    y1=table.col_values(1)
    x2=table.col_values(3)
    y2=table.col_values(4)
    #original point=(0,0)
    x3=table.col_values(6)
    y3=table.col_values(7)
    #sink=(10,10)
    x4=table.col_values(9)
    y4=table.col_values(10)
    plt.plot(x3, y3, '-')
    #plt.plot(x4, y4, 'g*',label=u"sink",size=10)
    plt.plot(5,10,'mD',label=u"sink=(5,10)")
    plt.plot(x1, y1, 'b->',label=lable1,linewidth=2)
    plt.plot(x2, y2, 'r->',label=lable2,linewidth=2)
    plt.grid(True)
    plt.xlabel(u'X-axis',size=18)
    plt.ylabel(u"Y-axis",size=18)
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
    plt.savefig("./figures/trajectory1_3.pdf",format='pdf', dpi=1000, bbox_inches = 'tight')
    plt.show()


def plot(fn,lable1,lable2,function):
    if function=="rewards":
        rewards(fn,lable1,lable2)
    elif function=="actorloss":
        actorloss(FN,lable1,lable2)
    elif function=="criticloss":
        criticloss(FN,lable1,lable2)
    elif function=="trajectory":
        trajectory(FN,lable1,lable2)

FN=r'C:/Users/dell/Desktop/results/last/No.7.xlsx'
lable1=u"$l_{0}$=(10,5),δ=0.1"
lable2=u"$l_{0}$=(10,5),δ=0.1"
plot(FN,lable1,lable2,"rewards")
plot(FN,lable1,lable2,"trajectory")
plot(FN,lable1,lable2,"actorloss")
#plot(FN,lable1,lable2,"criticloss")
"""

"""
def trajectory(file_name,lable1,lable2):
    data = xlrd.open_workbook(file_name)
    table = data.sheets()[2]
    x1=table.col_values(0)
    y1=table.col_values(1)
    x2=table.col_values(2)
    y2=table.col_values(3)
    x=[0,0,5,4,3,6,5,4]
    y=[0,10,5,6,3,5,4,5]
    area=[100,600,800,600,300,6000,800,600]
    m=['b','b','r','r','r','r','r','r']
    plt.scatter(x,y,s=area,c=m,alpha=0.6)
    for i in range(50):
        ex=np.random.choice([0,1,2,3,4,5,6,7,8,9])
        ey=np.random.choice([0,1,2,3,4,5,6,7,8,9])
        plt.plot(ex,ey,'x')
    #plt.scatter(ex,ey,s=area,c=m,alpha=0.6,marker='+')
    
    plt.plot(x1, y1, 'b--',label=u"trajectory($l_{0}$(5,0))",linewidth=2)
    plt.plot(x2, y2, 'r--',label=u"trajectory($l_{0}$(0,0))",linewidth=2)
    plt.plot(6,5,'r*',label=u"best_point",markersize=15)
    plt.plot(10,10,'b*',markersize=15)
    plt.plot(10,10,'g8',label=u"data sink",markersize=14,alpha=0.8)
    plt.plot(8,8,'mx',label=u"IoT device",markersize=8)
    plt.grid(True)
    plt.xlabel(u'X-axis',size=18)
    plt.ylabel(u"Y-axis",size=18)
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
    plt.savefig("./figures/trajectory1_multi_energy.pdf",format='pdf', dpi=1000, bbox_inches = 'tight')
    plt.show()

FN=r'C:/Users/dell/Desktop/muilty_energy/tra_multi_energy.xlsx'
lable1=u"$l_{0}$=(5,0),best_point=(10,10)"
lable2=u"$l_{0}$=(0,0),best_point=(5,5)"
trajectory(FN,lable1,lable2)

"""

def trajectory(file_name,lable1,lable2):
    data = xlrd.open_workbook(file_name)
    table = data.sheets()[2]
    x1=table.col_values(5)
    y1=table.col_values(6)
    x2=table.col_values(7)
    y2=table.col_values(8)
    x=[0,0,5,4.5,3,5,5,4]
    y=[0,10,5,6,3,6,4,5]
    area=[100,600,800,1000,300,10000,800,600]
    m=['b','b','r','r','r','r','r','r']
    plt.scatter(x,y,s=area,c=m,alpha=0.6)
    for i in range(50):
        ex=np.random.choice([0,1,2,3,4,5,6,7,8,9])
        ey=np.random.choice([0,1,2,3,4,5,6,7,8,9])
        plt.plot(ex,ey,'x')
    #plt.scatter(ex,ey,s=area,c=m,alpha=0.6,marker='+')
    
    plt.plot(x1, y1, 'b--',label=u"trajectory($l_{0}$(10,0))",linewidth=2)
    plt.plot(x2, y2, 'r--',label=u"trajectory($l_{0}$(10,0))",linewidth=2)
    plt.plot(5,6,'r*',label=u"best_point",markersize=15)
    plt.plot(5,10,'b*',markersize=15)
    plt.plot(5,10,'g8',label=u"data sink",markersize=14,alpha=0.8)
    plt.plot(8,8,'mx',label=u"IoT device",markersize=8)
    plt.grid(True)
    plt.xlabel(u'X-axis',size=18)
    plt.ylabel(u"Y-axis",size=18)
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
    plt.savefig("./figures/trajectory1_multi_energy2.pdf",format='pdf', dpi=1000, bbox_inches = 'tight')
    plt.show()

FN=r'C:/Users/dell/Desktop/muilty_energy/tra_multi_energy.xlsx'
lable1=u"$l_{0}$=(5,0),best_point=(10,10)"
lable2=u"$l_{0}$=(0,0),best_point=(5,5)"
trajectory(FN,lable1,lable2)
