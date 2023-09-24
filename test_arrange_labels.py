import numpy as np
from evaluate_model import arrange_labels

labels = np.random.rand(3, 5)
print('original labels: \n', labels)
original_classes = [1,2,3,4,5]
scored_classes = [{1}, {2,9}, {7}, {5}]
arranged_labels = arrange_labels(labels, original_classes, scored_classes)

print('arranged_labels: \n', arranged_labels)