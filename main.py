import os
import sys
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool


# np.random.seed(2018)
# random.seed(2018)
# tf.set_random_seed(2017)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_folder = 'D:/OneDrive - mail.ustc.edu.cn/PythonProjects/SGL/'
    else:
        root_folder = '/data0/hanlei/SGL_adj/'
    conf = Configurator(root_folder + "NeuRec.properties", default_section="hyperparameters")
    seed = conf["seed"]
    print('seed=', seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]

    dataset = Dataset(conf)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]
    with tf.Session(config=config) as sess:
        if importlib.util.find_spec("model.general_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.general_recommender." + recommender)
            
        elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            
            my_module = importlib.import_module("model.social_recommender." + recommender)
            
        else:
            my_module = importlib.import_module("model.sequential_rec ommender." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
        #######
#2022-07-27 20:13:32.299: epoch 23:	0.02814556  	0.06250791  	0.05154903  	0.09596046  	0.10759321
#2022-07-27 21:54:39.249: epoch 21:	0.03017105  	0.06718332  	0.05527309  	0.10128574  	0.11391312  warm_up=10 thresh=0.8
#2022-07-28 09:14:29.936: epoch 21:	0.03022316  	0.06728890  	0.05538537  	0.10143234  	0.11411693  warm_up=10 thresh=0.9
#2022-07-28 09:57:48.857: epoch 21:	0.03023418  	0.06733751  	0.05540871  	0.10148470  	0.11413421  warm_up=15 thresh=0.9
#2022-07-28 10:39:09.558: epoch 21:	0.03023102  	0.06731872  	0.05540148  	0.10147560  	0.11412214  warm_up=15 thresh=0.95
#2022-07-28 16:07:43.343: epoch 21:	0.03023102  	0.06732401  	0.05540348  	0.10147841  	0.11412617  warm_up=15 thresh=0.99
#2022-07-28 19:35:18.622: epoch 21:	0.03016787  	0.06725049  	0.05539227  	0.10181560  	0.11441138  warm_up=15 0.1-0.95
