# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 10:47:36 2022

@author: dell
"""
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
fig, ax = plt.subplots(2, 2)
ax = ax.flatten()

np.random.seed(0)
for i in range(4):
    weight = np.random.random([4, 4])
    im = ax[i].imshow(weight)


fig.colorbar(im, ax=[ax[0], ax[1], ax[2], ax[3]], fraction=0.06, pad=0.05)
plt.savefig('tjn.png', bbox_inches='tight')
plt.show()
"""
from matplotlib import pyplot as plt
import numpy as np

'''
颜色的选择：
'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 
'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 
'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 
'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 
'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 
'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 
'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 
 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 
 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 
 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 
 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 
 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 
 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 
 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 
 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 
 'twilight', 'twilight_r','twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

'''


def draw():
    # 定义热图的横纵坐标
    xLabel = np.arange(0,10,0.5)
    yLabel = np.arange(9.5,-0.1,-0.5)
    X=np.arange(0,10,0.5)
    Y=np.arange(9.5,-0.1,-0.5)
    X,Y=np.meshgrid(X,Y)
    # 准备数据阶段
    
    d1=(np.sqrt((X-5)**2+(Y-5)**2)+1)**2
    d2=(np.sqrt((X-4)**2+(Y-6)**2)+1)**2
    d22=(np.sqrt((X-4.5)**2+(Y-6)**2)+1)**2
    d3=(np.sqrt((X-3)**2+(Y-3)**2)+1)**2
    d4=(np.sqrt((X-6)**2+(Y-5)**2)+1)**2
    d44=(np.sqrt((X-5)**2+(Y-6)**2)+1)**2
    d5=(np.sqrt((X-5)**2+(Y-4)**2)+1)**2
    d6=(np.sqrt((X-4)**2+(Y-5)**2)+1)**2
    
    d7=(np.sqrt(X**2+Y**2)+1)**2
    d8=(np.sqrt((X-0)**2+(Y-10)**2)+1)**2
    r1=2*(30/d7+80/d8)/((np.sqrt((X-10)**2+(Y-10)**2)+1)**2)
    r2=2*(30/d7+80/d8)/((np.sqrt((X-5)**2+(Y-10)**2)+1)**2)
    r3=(100/d1+80/d22+50/d3+2600/d44+100/d5+80/d6)/((np.sqrt((X-5)**2+(Y-10)**2)+1)**2)
    r4=(100/d1+80/d2+50/d3+2000/d4+100/d5+80/d6)/((np.sqrt((X-10)**2+(Y-10)**2)+1)**2)

    # 作图阶段
    fig, ax = plt.subplots(2, 2, figsize=(15, 12), dpi=100)
    # 定义横纵坐标的刻度
    # ax1 = ax[0][0], ax2 = ax[0][1], ax3 = ax[1][0], ax4 = ax[1][1]

    # ax1
    ax[0][0].set_yticks(range(len(yLabel)))
    ax[0][0].set_yticklabels(yLabel)
    ax[0][0].set_xticks(range(len(xLabel)))
    ax[0][0].set_xticklabels(xLabel)
    im_25 = ax[0][0].imshow(r4)
    fig.colorbar(im_25, ax=ax[0][0])

    # ax2
    ax[0][1].set_yticks(range(len(yLabel)))
    ax[0][1].set_yticklabels(yLabel)
    ax[0][1].set_xticks(range(len(xLabel)))
    ax[0][1].set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择YlGn
    im_50 = ax[0][1].imshow(r1)
    fig.colorbar(im_50, ax=ax[0][1])

    # ax3
    ax[1][0].set_yticks(range(len(yLabel)))
    ax[1][0].set_yticklabels(yLabel)
    ax[1][0].set_xticks(range(len(xLabel)))
    ax[1][0].set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择YlGn
    im_75 = ax[1][0].imshow(r3)
    # 增加右侧的颜色刻度条
    # plt.colorbar(im_75)
    fig.colorbar(im_75, ax=ax[1][0])

    # ax4
    ax[1][1].set_yticks(range(len(yLabel)))
    ax[1][1].set_yticklabels(yLabel)
    ax[1][1].set_xticks(range(len(xLabel)))
    ax[1][1].set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择YlGn
    im_100 = ax[1][1].imshow(r2)
    # 增加右侧的颜色刻度条
    # plt.colorbar(im_100)
    fig.colorbar(im_100, ax=ax[1][1])

    # show
    # fig.colorbar(im_100, ax=ax.ravel().tolist())
    fig.tight_layout()
    plt.savefig('energy_hot.pdf')
    plt.show()


d = draw()

