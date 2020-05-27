# hacky hacky hacky method of quickly looking for and counting the selectivity
import numpy as np

in_data = np.load('allInputDataCurrent.npy')
out_data = np.load('allOutputDataCurrent.npy')

current_class = out_data[-1]
current_cluster = []

# as this is small data we can simply do...
data_on_neurons = in_data.T
# now rows are for each neuron, columns for each data point
# we know that we have 50 data points per class
clusters = []
for offset in range(0,500,50):
    indices = range(offset, 50 + offset)
    clusters.append(data_on_neurons[indices])

class_list = out_data[range(0,500,50)]

for n in range(len(AllDks[l])):
    print('currently on neuron:{0}p{1}'.format(n))
    #def brute_force_selectivity

foundSelectivityList = []
foundNeuronList = []

n=3
offsets = range(0,500,50)

for n in range(500):
    for class_r in range(10):
        matches_range = range(offsets[class_r], 50 + offsets[class_r])
        notmacthes_range = [x for x in range(500) if not x in matches_range]
       # notmatches = data_on_neurons[n][range(offsets[class_c], 50 + offsets[class_c])]
        matches = data_on_neurons[n][matches_range]
        notmatches = data_on_neurons[n][notmacthes_range]
        minMatch, maxMatch = min(matches), max(matches)
        minNotMatch, maxNotMatch = min(notmatches), max(notmatches)
        #print('{0} matches: from {1} to {2}'.format(len(matches), minMatch, maxMatch))
        #print('{0} matches: from {1} to {2}'.format(len(notmatches), minNotMatch, maxNotMatch))
        if maxMatch <= minNotMatch:
            left = matches
            right = notmatches
        elif maxNotMatch <= minMatch:
            left = notmatches
            right = matches
        selectivity = min(right) - max(left)
        print('Selectivity of {0}'.format(selectivity))
        if abs(selectivity) > 0.01:
            print('Selecitivy of {} found!'.format(selectivity))
            print('Neuron no {}'.format(n))
            print('Class no {}'.format(class_r))
            # print('current_key is: {}'.format(current_key))
            foundSelectivityList.append(selectivity)
            foundNeuronList.append([n])



for neuron_no in range(len(in_data)):

    class_for_this_neuron = out_data[neuron_no]
    input_for_this_neuron = in_data[neuron_no]
    if not (class_for_this_neuron == current_class).all():
        current_cluster = []




