import torch
import numpy as np

# Each key represents a valid combination of movement, speed, and direction
# The value is the class label #
'''
_condense_dict = {
    (0, 0, 0): 0,
    (1, 1, 1): 1,
    (1, 1, 2): 2,
    (1, 2, 1): 3,
    (1, 2, 2): 4
}
'''
'''
0 - no movement
1 - movement (slow, up/right)
2 - movement (slow, down/left)
3 - movement (fast, up/right)
4 - movement( fast, down/left)
'''
_condense_dict5 = {
    (0, 0, 0): 0,
    (1, 1, 1): 1,
    (1, 1, 2): 2,
    (1, 2, 1): 3,
    (1, 2, 2): 4
}

# Each key represents a valid combination of movement, speed, and direction
# The value is the class label 3 class problem for direction
_condense_dict3 = {
    (0, 0, 0): 0,
    (1, 1, 1): 1,
    (1, 2, 1): 1,
    (1, 1, 2): 2,
    (1, 2, 2): 2
}

'''
_speed_dect = {
    (0, 0, 0): 0,
    (1, 1, 1): 1,
    (1, 2, 1): 1,
    (1, 1, 2): 2,
    (1, 2, 2): 2
}
'''

# Used to compare 1 hot encoded input to integer target value
def is_correct_label(inputs, target):
    return [torch.equal(torch.argmax(x, 0), y) for (x, y) in zip(inputs, target)]


def avg(lst):
    return sum(lst) / len(lst)


# Convert 3 movement, direction, speed labels into a single label. Return None if the combination is invalid
'''
def condense_outputs(gt, invalid_value=-1):
    ret_val = np.zeros(gt.shape[1],)
    for i in range(gt.shape[1]):
        curr_key = tuple(gt[:, i])
        ret_val[i] = _condense_dict[curr_key] if curr_key in _condense_dict else invalid_value
    return ret_val
'''

def condense_outputs3class(gt, invalid_value=-1):
    ret_val = np.zeros(gt.shape[1],)
    for i in range(gt.shape[1]):
        curr_key = tuple(gt[:, i])
        ret_val[i] = _condense_dict3[curr_key] if curr_key in _condense_dict3 else invalid_value
    return ret_val

def condense_outputs5class(gt, invalid_value=-1):
    ret_val = np.zeros(gt.shape[1],)
    for i in range(gt.shape[1]):
        curr_key = tuple(gt[:, i])
        ret_val[i] = _condense_dict5[curr_key] if curr_key in _condense_dict5 else invalid_value
    return ret_val


# Just base the output on the presence of movement
# Don't consider direction or speed
def condense_outputs_movement(gt):
    ret_val = np.zeros(gt.shape[1], )
    for i in range(gt.shape[1]):
        curr_key = tuple(gt[:, i])
        curr_gt = gt[:, i]
        ret_val[i] = curr_gt[0]
    return ret_val
