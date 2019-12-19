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
            axes.legend((real, fake, pc),('Real_Images, Mean:  ' + str(real_mean) + ' $\pm$ ' + str(real_std),
                                            'Fake_Images, Mean:  ' + str(fake_mean) + ' $\pm$ ' + str(fake_std),
                                            'Pearson Corr:' + str(summary_dict[i]['corr'])))
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

if __name__ == '__main__':
    ###
    evaldir = './../Heart'

    nr_folds = 2
    mode= 'test'
    # mode='train'
    # mode= 'validation'

    coll_plot()