import os
import numpy as np
import pandas as pd
import process_limited
import random
def calculate_ci(mre,len):
    mre[:, len+1] = np.mean(mre[:, 0:len], axis=1)
    return mre
def system_samplesize(sys_name):
    if (sys_name == 'Apache'):
        N_train_all = np.multiply(9, [1, 2, 4, 6])  # This is for Apache
    elif (sys_name == 'BDBJ'):
        N_train_all = np.multiply(26, [1, 2, 4, 6])  # This is for BDBJ
    elif (sys_name == 'BDBC'):
        N_train_all = np.multiply(18, [1, 2, 4, 6])  # This is for BDBC
    elif (sys_name == 'LLVM'):
        N_train_all = np.multiply(11, [1, 2, 4, 6])  # This is for LLVM
    elif (sys_name == 'SQL'):
        N_train_all = np.multiply(39, [1, 2, 4, 6])  # This is for SQL
    elif (sys_name == 'x264'):
        N_train_all = np.multiply(16, [1, 2, 4, 6])  # This is for X264
    elif (sys_name == 'WGet'):
        N_train_all = np.multiply(16, [1, 2, 4, 6])  # This is for WGet
    elif (sys_name == 'vp9'):
        N_train_all = np.multiply(42, [1, 2, 4, 6])  # This is for vp9
    elif (sys_name == 'polly'):
        N_train_all = np.multiply(40, [1, 2, 4, 6])  # This is for polly
    elif (sys_name == 'lrzip'):
        N_train_all = np.multiply(19, [1, 2, 4, 6])  # This is for lrzip
    elif (sys_name == 'Dune'):
        N_train_all = np.asarray([49, 78, 384, 600])  # This is for Dune
    elif (sys_name == 'hipacc'):
        N_train_all = np.asarray([261, 528, 736, 1281])  # This is for hipacc
    elif (sys_name == 'hsmgp'):
        N_train_all = np.asarray([77, 173, 384, 480])  # This is for hsmgp
    elif (sys_name == 'javagc'):
        N_train_all = np.asarray([855, 2571, 3032, 5312])  # This is for javagc
    elif (sys_name == 'sac'):
        N_train_all = np.asarray([2060, 2295, 2499, 3261])  # This is for sac
    elif (sys_name == 'LLVM'):
        N_train_all = np.multiply(11, [1, 2, 4, 6])  # This is for LLVM
    else:
        raise AssertionError("Unexpected value of 'sys_name'!")

    return N_train_all

def transferdata(config):
    min_vals = config.min(axis=0)
    max_vals = config.max(axis=0)
    range_vals = max_vals - min_vals

    range_vals[range_vals == 0] = 1
 
    normalized_config = (config - min_vals) / range_vals

    return normalized_config

if __name__ == "__main__":

    sys_name = "x264"
    N_train_all = system_samplesize(sys_name)
    result_file_path = "./datasets/" + sys_name + "_AllNumeric.csv"
    all_input_signal = pd.read_csv(result_file_path, index_col=0)
    all_input_signal = np.asarray(all_input_signal)
    config = np.delete(all_input_signal, -1, 1)
    config=transferdata(config)

    repeat_times = 20

    mre_result = np.zeros((len(N_train_all), 22), dtype=float)
    init_length = 6  # modify the length of a random initial selection
    stop_point = N_train_all[-1] -init_length # modify stop point
    for i in range(0, repeat_times):
        print('seed:' + str(i))
        np.random.seed(i)
        init_set = list(np.random.randint(len(all_input_signal), size=init_length))

        sampled_config_ids, mre_result = process_limited.hsbal(N_train_all, config, init_set, all_input_signal, i,
                                                                    stop_point, mre_result)

        mre_result = calculate_ci(mre_result, repeat_times)
        df = pd.DataFrame(mre_result)
        df.to_csv('./result/'+sys_name+'_1mre_result.csv')

