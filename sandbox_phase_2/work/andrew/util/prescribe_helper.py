
import numpy as np
from .preprocess import get_all_data
from .write_solutions import write_solutions
from .trainer import Trainer
import tensorflow as tf
import pandas as pd
import os
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')

NB_ACTION = 12
MAX_NPI_SUM = 34

NPI_COLUMNS = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']

def prescribe_help(start_date, end_date, ips_path, weights_path, output_csv):
    
    num_days = int(1 + (np.datetime64(end_date)-np.datetime64(start_date))/(np.timedelta64(1, 'D')))
    
    t = Trainer(num_days)
    # todo - handle rolling out data if start date is in the future

    data = get_all_data(start_date, DATA_FILE_PATH, ips_path, weights_path)
    
    geos = data['geos']
    
    dummy_df = pd.DataFrame(columns=NPI_COLUMNS + ["Date", "CountryName", "RegionName", "PrescriptionIndex"])
    dummy_df.to_csv(output_csv, index=False)
    
    for g in geos:
        print("prescribing for", g)
        s = time.time()
        g_inputs = data['input_tensors'][g]
        country_name, region_name = data['country_names'][g]
        #country_name = df[df.GeoID == g].iloc[0]["CountryName"]
        #region_name = df[df.GeoID == g].iloc[0]["RegionName"]
        
        prescriptions = prescribe_one_geo(t, g_inputs)
        e = time.time()
        print(e-s, "seconds to prescribe")
        write_solutions(prescriptions, country_name, region_name, start_date, num_days, output_csv)


def convert_weights(weights):
    return tf.reshape(weights, (-1, 12))
        
# get rid of stringency calc & see how it runs
# schedule gradient application
def prescribe_one_geo(t, inputs):
    t.set_inputs_and_goal(inputs)

    total_updates = t.num_days * 34
    
    thresholds = np.linspace(0.90, 0.1, 10)
    current_threshold_idx = 0
    
    weights_list = []
    
    
    
    update_stride = tf.cast(tf.math.floor(tf.math.divide(total_updates, 125)), 'int32')
    update_stride = tf.math.maximum(update_stride, 1)
    # am i dividing incorrectly for stringency?
    for i in range(125):
        c, u = t.train_loop(t.inputs, t.min_cases, t.npi_weights, update_stride)
        p = tf.math.divide(c, t.max_reduction)
        
        #print(c, p, t.max_reduction)
        #print(t.min_cases, t.max_reduction, c)
        if p < thresholds[current_threshold_idx]:
            #print(f"found solution at {tf.constant(i)} with performance {p.numpy()[0]} and stringency {s}")
            
            weights_list.append(tf.constant(tf.reshape(t.prescriptor.trainable_weights, (-1, NB_ACTION))))
            #weights_list.append(convert_weights(t.prescriptor.trainable_weights))
            current_threshold_idx += 1
            
        if current_threshold_idx == 10:
            #print("found 10 solutions")
            break
        #weights_list.append(convert_weights(t.prescriptor.trainable_weights))
        #percentages.append(p)
        #stringencies.append(s)
        #weights_list.append(list(map(lambda x: tf.constant(x), t.prescriptor.trainable_weights)))
    
    if len(weights_list) < 10:
        print("fewer than 10 solutions found")
        for i in range(10-len(weights_list)):
            weights_list.append(tf.zeros((t.num_days, NB_ACTION)))
    return weights_list