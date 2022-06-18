import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_results(item,ylabel='rewards_train', save_fig = True):
    '''plot using searborn to plot
    '''
    sns.set()
    plt.figure()
    plt.plot(np.arange(len(item)), item)
    plt.title(ylabel+' of DDPG')
    plot_results(rewards)
    plot_results(moving_average_rewards,ylabel='moving_average_rewards_'+tag)
    plt.ylabel(ylabel)
    plt.xlabel('episodes')
    if save_fig:
        plt.savefig(os.path.dirname(__file__)+"/result/"+ylabel+".png")
    plt.show()

if __name__ == "__main__":

    output_path = os.path.split(os.path.abspath(__file__))[0]+"/result/20210609-173655/"
    print(output_path)
    tag = 'train'
    rewards=np.load(output_path+"rewards_"+tag+".npy", )
    print(rewards)
    moving_average_rewards=np.load(output_path+"moving_average_rewards_"+tag+".npy",)
    steps=np.load(output_path+"steps_"+tag+".npy")
    
    plot_results(steps,ylabel='steps_'+tag)
    tag = 'eval'
    rewards=np.load(output_path+"rewards_"+tag+".npy", )
    moving_average_rewards=np.load(output_path+"moving_average_rewards_"+tag+".npy",)
    steps=np.load(output_path+"steps_"+tag+".npy")
    plot_results(rewards,ylabel='rewards_'+tag)
    plot_results(moving_average_rewards,ylabel='moving_average_rewards_'+tag)
    plot_results(steps,ylabel='steps_'+tag)