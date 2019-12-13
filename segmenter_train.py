# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

from switch.data_switch import data_switch
from segmenter.model_segmenter import segmenter
from segmenter.experiments import heart_config as exp_config

def main(cv_fold):

    exp_config.cv_fold = cv_fold

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config.data_root, exp_config.cv_fold)

    # Build model
    segmenter_model = segmenter(exp_config=exp_config, data=data, fixed_batch_size=exp_config.batch_size)

    # Train model
    segmenter_model.train()


if __name__ == '__main__':

    main(exp_config.cv_fold)
