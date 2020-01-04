import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import glob
import pickle


"""
structure of summary_dict: 

-   1st param:   0: sample volume 0
                 1: sample volume 1  (until you reach nr_samples)
                'nr_samples': how many sample volumes
                
-   2nd params: 'dice': dice score
                'all_dice': non-averaged dice scores
                'corr': pearson correlation 
                'real_frac': real collagen fraction 
                'fake_frac': fake collagen fraction 
                'sample_idx': which samples are in which set for which fold
                
"""



def coll_plot():

    # curr_path = os.path.join(evaldir + '/summary_dicts_fold' + str(1) + '.p')

    for fold in range(4, 5): #(1, nr_folds)
        if mode == 'test':
            curr_path = os.path.join(evaldir + '/test_summary_dicts_fold' + str(fold) + '.p')
        elif mode == 'train':
            curr_path = os.path.join(evaldir + '/train_summary_dicts_fold' + str(fold) + '.p')
        elif mode == 'validation':
            curr_path = os.path.join(evaldir + '/validation_summary_dicts_fold' + str(fold) + '.p')

        with open(curr_path, 'rb') as file:
            summary_dict = pickle.load(file)

        for i in range(summary_dict['nr_samples']):
            real_frac = np.asarray(summary_dict[i]['real_frac'])
            fake_frac = np.asarray(summary_dict[i]['fake_frac'])

            real_mean = np.around(np.mean(real_frac), decimals=3)
            real_std = np.around(np.std(real_frac), decimals=3)
            fake_mean = np.around(np.mean(fake_frac), decimals=3)
            fake_std = np.around(np.std(fake_frac), decimals=3)

            fig, axes = plt.subplots(1, 1)

            real, = axes.plot(np.arange(len(real_frac)), real_frac, '*b', alpha=0.5)
            fake, = axes.plot(np.arange(len(real_frac)), fake_frac, '*r', alpha=0.5)
            pc, = axes.plot(0,0, linestyle="None")
            dice, = axes.plot(0, 0, linestyle='None')

            axes.legend((real, fake, pc, dice),('Real_Images, Mean:  ' + str(real_mean) + ' $\pm$ ' + str(real_std),
                                            'Fake_Images, Mean:  ' + str(fake_mean) + ' $\pm$ ' + str(fake_std),
                                            'Pearson Corr:' + str(summary_dict[i]['corr']),
                                          'Dice Score:' + str(summary_dict[i]['dice'])))
            if mode == 'test':
                axes.set_title('Fold ' + str(fold) + ', Test Volume: ' + str(summary_dict[i]['sample_idx']))
            elif mode == 'train':
                axes.set_title('Fold ' + str(fold) + ', Training Volume: ' + str(summary_dict[i]['sample_idx']))
            elif mode == 'validation':
                axes.set_title('Fold ' + str(fold) + ', Validation Volume: ' + str(summary_dict[i]['sample_idx']))

            axes.set_ylim([0,0.1])
            axes.set_xlabel('Slice Nr.')
            axes.set_ylabel('Collagen Ratio ('r'$\frac{collagen}{cells}$)')

            plt.show()

def bar_plt_indiv(directory, sample_nr):

    files = os.listdir(directory)
    metric_dict = {}
    std_dev_dict = {}

    help_idx = 0
    file_list = []
    for f in files:
        print(os.path.join(directory, f))
        with open((directory + f), 'rb') as file:
            summary_dict = pickle.load(file, encoding='bytes') #use this encoding so python3 can work with python 2 pickle files (from UDT)
            file_list.append(os.path.splitext(f)[0])

        dice = np.around(np.asarray(summary_dict[sample_nr]['dice']), decimals=3)

        real_frac = np.asarray(summary_dict[sample_nr]['real_frac'])
        fake_frac = np.asarray(summary_dict[sample_nr]['fake_frac'])

        real_mean = np.around(np.mean(real_frac), decimals=3)
        real_std = np.around(np.std(real_frac), decimals=3)
        fake_mean = np.around(np.mean(fake_frac), decimals=3)
        fake_std = np.around(np.std(fake_frac), decimals=3)

        abs_diff = np.around(np.sum(np.absolute((real_frac - fake_frac)/ real_frac.shape[0])),decimals=3)

        metric_dict[help_idx] = [dice, real_mean, fake_mean, abs_diff]
        std_dev_dict[help_idx] = [0, real_std, fake_std, 0]

        help_idx += 1

    labels = ['Dice Score', 'Mean Frac. Real', 'Mean Frac. Pred.', 'Abs. Diff.']

    x = np.arange(len(labels))  # the label locations
    width = 0.18  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, metric_dict[0], width, yerr=std_dev_dict[0],label=file_list[0])
    rects2 = ax.bar(x + width / 2, metric_dict[1], width, yerr=std_dev_dict[1], label=file_list[1])
    rects3 = ax.bar(x + width * 3 / 2, metric_dict[2], width, yerr=std_dev_dict[2], label=file_list[2])
    rects4 = ax.bar(x + width * 5 / 2, metric_dict[3], width, yerr=std_dev_dict[3], label=file_list[3])
    rects5 = ax.bar(x + width * 7 / 2, metric_dict[4], width, yerr=std_dev_dict[4], label=file_list[4])

    x_ticks = x + width * 3 / 2

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Segmentation Evaluation Metrics')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()

    plt.show()
    fig.savefig('./../Images/Vol_' + str(sample_nr) + '.eps', format='eps')

def bar_plt_avg(directory):

    files = os.listdir(directory)
    metric_dict = {}
    std_dev_dict = {}

    help_idx = 0
    file_list = []
    for f in files:
        print(os.path.join(directory, f))
        with open((directory + f), 'rb') as file:
            summary_dict = pickle.load(file, encoding='bytes') #use this encoding so python3 can work with python 2 pickle files (from UDT)
            file_list.append(os.path.splitext(f)[0])

        dice = np.around(np.asarray(summary_dict[sample_nr]['dice']), decimals=3)

        real_frac = np.asarray(summary_dict[sample_nr]['real_frac'])
        fake_frac = np.asarray(summary_dict[sample_nr]['fake_frac'])

        real_mean = np.around(np.mean(real_frac), decimals=3)
        real_std = np.around(np.std(real_frac), decimals=3)
        fake_mean = np.around(np.mean(fake_frac), decimals=3)
        fake_std = np.around(np.std(fake_frac), decimals=3)

        abs_diff = np.around(np.sum(np.absolute((real_frac - fake_frac)/ real_frac.shape[0])),decimals=3)

        metric_dict[help_idx] = [dice, real_mean, fake_mean, abs_diff]
        std_dev_dict[help_idx] = [0, real_std, fake_std, 0]

        help_idx += 1

    labels = ['Dice Score', 'Mean Frac. Real', 'Mean Frac. Pred.', 'Abs. Diff.']

    x = np.arange(len(labels))  # the label locations
    width = 0.18  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, metric_dict[0], width, yerr=std_dev_dict[0],label=file_list[0])
    rects2 = ax.bar(x + width / 2, metric_dict[1], width, yerr=std_dev_dict[1], label=file_list[1])
    rects3 = ax.bar(x + width * 3 / 2, metric_dict[2], width, yerr=std_dev_dict[2], label=file_list[2])
    rects4 = ax.bar(x + width * 5 / 2, metric_dict[3], width, yerr=std_dev_dict[3], label=file_list[3])
    rects5 = ax.bar(x + width * 7 / 2, metric_dict[4], width, yerr=std_dev_dict[4], label=file_list[4])

    x_ticks = x + width * 3 / 2

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Segmentation Evaluation Metrics')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()

    plt.show()
    fig.savefig('./../Images/Vol_' + str(sample_nr) + '.eps', format='eps')

if __name__ == '__main__':
    ###
    evaldir = './../Heart'
    # evaldir = './../Heart_limited'

    nr_folds = 2
    mode= 'test'
    # mode='train'
    # mode= 'validation'

    # coll_plot()

    bar_dir = './../Heart_all/'
    # bar_dir = './../Heart_all_no_aug/'
    sample_nr = 0
    bar_plt_indiv(bar_dir, sample_nr)

    # bar_plt_avg(bar_dir)

