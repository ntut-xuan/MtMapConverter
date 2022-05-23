import numpy as np

labels = np.loadtxt("labels.txt", dtype=np.str_)
ID_table = np.loadtxt("ID.txt", dtype=np.uint8)
label_to_id = {}

id_index = 0

for i in range(len(labels)):
    if labels[i] not in label_to_id:
        print(labels[i], ID_table[id_index])
        label_to_id[labels[i]] = ID_table[id_index]
        id_index += 1
