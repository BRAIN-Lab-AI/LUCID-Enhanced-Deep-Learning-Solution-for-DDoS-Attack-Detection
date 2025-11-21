import numpy as np
import h5py
import glob
from collections import OrderedDict


SEED = 1
MAX_FLOW_LEN = 100 # number of packets
TIME_WINDOW = 10
TRAIN_SIZE = 0.90

protocols = ['arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','telnet','tcp','udp']
powers_of_two = np.array([2**i for i in range(len(protocols))])

feature_list = OrderedDict([
    ('timestamp', [0,10]),
    ('packet_length',[0,1<<16]),
    ('highest_layer',[0,1<<32]),
    ('IP_flags',[0,1<<16]),
    ('protocols',[0,1<<len(protocols)]),
    ('TCP_length',[0,1<<16]),
    ('TCP_ack',[0,1<<32]),
    ('TCP_flags',[0,1<<16]),
    ('TCP_window_size',[0,1<<16]),
    ('UDP_length',[0,1<<16]),
    ('ICMP_type',[0,1<<8])]
)


def load_dataset(path):
    # Use a glob to find the file matching the pattern
    try:
        filename = glob.glob(path)[0]
    except IndexError:
        print(f"Error: No file found matching path: {path}")
        return None, None
    

    with h5py.File(filename, "r") as f:
        set_x_orig = np.array(f["set_x"][:])  # features
        set_y_orig = np.array(f["set_y"][:])  # labels

    # Reshape features to add the channel dimension for the CNN
    X_train = np.reshape(set_x_orig, (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1))
    Y_train = set_y_orig

    return X_train, Y_train

def normalize_and_padding(X, mins, maxs, max_flow_len):

    normalized_and_padded_data = []

    if not X:
        return np.array([])

    # Determine the number of features from the first fragment
    num_features = X[0].shape[1]
    
    # Calculate the denominator once to avoid recalculating in the loop
    denominator = maxs - mins
    
    for fragment in X:
        # Use np.divide for safe division. Where the denominator is 0, the result will be 0.
        normalized_fragment = np.divide(
            (fragment - mins), 
            denominator, 
            out=np.zeros_like(fragment, dtype=float), # Create an output array of zeros
            where=(denominator != 0) # Only perform division where the denominator is not zero
        )
        
        current_len = normalized_fragment.shape[0]
        
        # Pad or truncate the fragment to match max_flow_len
        if current_len < max_flow_len:
            padding_needed = max_flow_len - current_len
            padding = np.zeros((padding_needed, num_features))
            padded_fragment = np.vstack((normalized_fragment, padding))
        else:
            padded_fragment = normalized_fragment[:max_flow_len, :]
            
        normalized_and_padded_data.append(padded_fragment)
        
    # This final conversion creates a clean, uniform NumPy array
    return np.array(normalized_and_padded_data)



def count_packets_in_dataset(X_list):
    packet_counters = []
    for X in X_list:
        TOT = X.sum(axis=2)
        packet_counters.append(np.count_nonzero(TOT))
    return packet_counters

def static_min_max(time_window=10):
    feature_list['timestamp'][1] = time_window
    min_array = np.zeros(len(feature_list))
    max_array = np.zeros(len(feature_list))
    i=0
    for feature, value in feature_list.items():
        min_array[i] = value[0]
        max_array[i] = value[1]
        i+=1
    return min_array,max_array

