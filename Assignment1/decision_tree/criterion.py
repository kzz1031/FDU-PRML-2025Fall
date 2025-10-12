"""
criterion
"""
import math

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    info_gain = 0.0
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 计算父节点熵
    total_samples = sum(all_labels.values())
    parent_entropy = 0.0
    for count in all_labels.values():
        if count > 0:
            p = count / total_samples
            parent_entropy -= p * math.log2(p)
    
    # 计算左子节点熵
    left_samples = sum(left_labels.values())
    left_entropy = 0.0
    if left_samples > 0:
        for count in left_labels.values():
            if count > 0:
                p = count / left_samples
                left_entropy -= p * math.log2(p)
    
    # 计算右子节点熵
    right_samples = sum(right_labels.values())
    right_entropy = 0.0
    if right_samples > 0:
        for count in right_labels.values():
            if count > 0:
                p = count / right_samples
                right_entropy -= p * math.log2(p)
    
    # 计算加权子节点熵
    weighted_child_entropy = (left_samples / total_samples) * left_entropy + (right_samples / total_samples) * right_entropy
    info_gain = parent_entropy - weighted_child_entropy
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 计算分裂信息
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    total_samples = sum(all_labels.values())
    left_samples = sum(left_labels.values())
    right_samples = sum(right_labels.values())
    # 计算分裂信息
    split_info = 0.0
    if left_samples > 0:
        p_left = left_samples / total_samples
        split_info -= p_left * math.log2(p_left)
    if right_samples > 0:
        p_right = right_samples / total_samples
        split_info -= p_right * math.log2(p_right)
    # 计算信息增益率
    if split_info > 0:
        info_gain = info_gain / split_info
    else:
        info_gain = 0.0  
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 计算分裂前的Gini指数
    total_samples = sum(all_labels.values())
    before = 1.0
    for count in all_labels.values():
        if count > 0:
            p = count / total_samples
            before -= p * p
    
    # 计算分裂后的Gini指数
    left_samples = sum(left_labels.values())
    right_samples = sum(right_labels.values())
    
    left_gini = 1.0
    if left_samples > 0:
        for count in left_labels.values():
            if count > 0:
                p = count / left_samples
                left_gini -= p * p
    
    right_gini = 1.0
    if right_samples > 0:
        for count in right_labels.values():
            if count > 0:
                p = count / right_samples
                right_gini -= p * p
    
    after = (left_samples / total_samples) * left_gini + (right_samples / total_samples) * right_gini
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 计算分裂前的分类误差率
    total_samples = sum(all_labels.values())
    max_count = max(all_labels.values()) if all_labels else 0
    before = 1.0 - (max_count / total_samples) if total_samples > 0 else 0.0
    
    # 计算分裂后的分类误差率
    left_samples = sum(left_labels.values())
    right_samples = sum(right_labels.values())
    
    left_max_count = max(left_labels.values()) if left_labels else 0
    left_error = 1.0 - (left_max_count / left_samples) if left_samples > 0 else 0.0
    
    right_max_count = max(right_labels.values()) if right_labels else 0
    right_error = 1.0 - (right_max_count / right_samples) if right_samples > 0 else 0.0
    
    after = (left_samples / total_samples) * left_error + (right_samples / total_samples) * right_error
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
