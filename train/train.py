from klapeyron_py_utils.train.train import Train_Pipeline
from klapeyron_py_utils.data_pipe.data_manager import Data_manager
from data.preprocess import Pre
from klapeyron_py_utils.models.ResNet34_v2 import ResNet34_v2
from klapeyron_py_utils.models.ResNet14_v2 import ResNet14_v2
from klapeyron_py_utils.models.ResNet14_v2_mini import ResNet14_v2_mini
from klapeyron_py_utils.models.ResNet10_v2_mini import ResNet10_v2_mini
from klapeyron_py_utils.models.configs.model_config import Model_Config
from klapeyron_py_utils.models.configs.model_train_config import Model_Train_Config
from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow(3)
from data.csv import CSV_c

# csv_paths = ['D:/has_glasses_data/MeGlass/data.csv', 'D:/has_glasses_data/example_data_glasses/data.csv',]
csv_paths = ['D:/has_glasses_data/Celeba/data.csv']

size = 120
preproc_trn = Pre([Pre.READ_TF_IMG, Pre.FLIP, Pre.ROTATE, Pre.CROP_BB, Pre.RESIZE_PROPORTIONAL_TF_BILINEAR, Pre.STANDARDIZE, Pre.PAD_TO_INPUT_CENTER], size)
preproc_val = Pre([Pre.READ_TF_IMG, Pre.CROP_BB, Pre.RESIZE_PROPORTIONAL_TF_BILINEAR, Pre.STANDARDIZE, Pre.PAD_TO_INPUT_CENTER], size)

dm = Data_manager(256, csv_paths, preproc_trn, preproc_val, CSV_c.SAMPLE_FILE, CSV_c)
m_config = Model_Config((size,size,3))
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
m_trn_config = Model_Train_Config(optimizer, reg_l2_beta=0.01, dropout_drop_prob=0.1)
m = ResNet14_v2_mini(m_config, m_trn_config)
tr = Train_Pipeline('F:/has_glasses/exp6_run0', csv_paths, dm, m, resume_from_log=Train_Pipeline.LOG_TAG_BEST_EER)
tr.train()
