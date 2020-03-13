from data.csv import CSV_c
from klapeyron_py_utils.train.eval import Eval_Pipeline
from klapeyron_py_utils.data_pipe.data_manager import Data_manager
from data.preprocess import Pre

size = 120
preproc_trn = Pre([Pre.READ_TF_IMG, Pre.FLIP, Pre.ROTATE, Pre.CROP_BB, Pre.RESIZE_PROPORTIONAL_TF_BILINEAR, Pre.STANDARDIZE, Pre.PAD_TO_INPUT_CENTER], size)
preproc_val = Pre([Pre.READ_TF_IMG, Pre.CROP_BB, Pre.RESIZE_PROPORTIONAL_TF_BILINEAR, Pre.STANDARDIZE, Pre.PAD_TO_INPUT_CENTER], size)

csv_paths = ['D:/has_glasses_data/example_data_glasses/data.csv']
checkpoint_path = 'F:\has_glasses\exp6_run0\checkpoints\checkpoint_opt_eer'


print('final_test:')
dm = Data_manager(150, csv_paths, preproc_trn, preproc_val, CSV_c.SAMPLE_FILE, CSV_c, folds_to_eval=[CSV_c.FOLD_TST])
e = Eval_Pipeline(dm, checkpoint_path)
e.eval_val(logs_print=True)


print()
print('Celeba_test:')
csv_paths = ['D:/has_glasses_data/Celeba/data.csv']

dm = Data_manager(150, csv_paths, preproc_trn, preproc_val, CSV_c.SAMPLE_FILE, CSV_c, folds_to_eval=[CSV_c.FOLD_TST])
e = Eval_Pipeline(dm, checkpoint_path)
e.eval_val(logs_print=True)