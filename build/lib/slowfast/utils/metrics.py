#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch
import numpy as np
import pandas as pd
import os

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]



def multitask_topks_correct(preds, labels, ks=(1,)):
    """
    Args:
        preds: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        ks: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    all_correct = all_correct.to(preds[0].device)
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    multitask_topks_correct = [
        torch.ge(all_correct[:k].float().sum(0), task_count).float().sum(0) for k in ks
    ]

    return multitask_topks_correct


def multitask_topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
   """
    num_multitask_topks_correct = multitask_topks_correct(preds, labels, ks)
    return [(x / preds[0].size(0)) * 100.0 for x in num_multitask_topks_correct]

def MT5R(test_score_dict, cfg):
    """
    Args:
        test_score_dict: Contains the prediciton scores of verbs and nouns along with the labels
    Returns:
        Mean Top-5 Recall
    """
    verb_num_classes = 97
    noun_num_classes = 300
    action_num_classes = 3806
    verb_class_wise_frequency = np.zeros(verb_num_classes)
    verb_class_wise_correct5 = np.zeros(verb_num_classes)
    noun_class_wise_frequency = np.zeros(noun_num_classes)
    noun_class_wise_correct5 = np.zeros(noun_num_classes)
<<<<<<< HEAD
    action_class_wise_frequency = np.zeros(action_num_classes)
    action_class_wise_correct5 = np.zeros(action_num_classes)

    action_list = pd.read_csv(os.path.join(cfg.EPICKITCHENS.ANNOTATIONS_DIR, cfg.EPICKITCHENS.ACTIONS_LIST)) # id (action id), verb, noun, action 
    

=======
    action_class_wise_correct5 = np.zeros(action_num_classes)
>>>>>>> e0ef9a0442f6ba31ffe45ac06f6b3bf13782c7de
    for nid in test_score_dict.keys():
        verb_scores = [ score_dict['verb'] for score_dict in test_score_dict[nid] ]
        noun_scores = [ score_dict['noun'] for score_dict in test_score_dict[nid] ]
        verb_score = torch.mean(torch.stack(verb_scores), dim=0)
        noun_score = torch.mean(torch.stack(noun_scores), dim=0)
        _, verb_pred5 = verb_score.topk(k=5, dim=-1)
        _, noun_pred5 = noun_score.topk(k=5, dim=-1)

        verb_label = test_score_dict[nid][0]['verb_label']
        noun_label = test_score_dict[nid][0]['noun_label']
        action_label = test_score_dict[nid][0]['action_label']
        verb_target5 = verb_label.unsqueeze(dim=-1).expand_as(verb_pred5)
        noun_target5 = noun_label.unsqueeze(dim=-1).expand_as(noun_pred5)
        verb_correct5 = verb_target5[verb_pred5 == verb_target5] # [] if different and class_num if same
        noun_correct5 = noun_target5[noun_pred5 == noun_target5] # [] if different and class_num if same
<<<<<<< HEAD
    
        verb_cls = verb_label.item()
        verb_class_wise_frequency[verb_cls] += 1
        verb_class_wise_correct5[verb_cls] += (verb_correct5 == verb_cls).sum().item()
=======

        for cls in range(verb_num_classes):
            verb_class_wise_frequency[cls] += (verb_label == cls).sum().item()
            verb_class_wise_correct5[cls] += (verb_correct5 == cls).sum().item()
>>>>>>> e0ef9a0442f6ba31ffe45ac06f6b3bf13782c7de
        
        noun_cls = noun_label.item()
        noun_class_wise_frequency[noun_cls] += 1
        noun_class_wise_correct5[noun_cls] += (noun_correct5 == noun_cls).sum().item()

        # to compute mean action recall
        action_cls = action_label.item()
        verb_cls = int(action_list.loc[action_list['id'] == action_cls]['verb'].values)
        noun_cls = int(action_list.loc[action_list['id'] == action_cls]['noun'].values)
        if (verb_correct5 == verb_cls).sum().item() > 0 and (noun_correct5 == noun_cls).sum().item() > 0:
            action_class_wise_correct5[action_cls] += 1
        action_class_wise_frequency[action_cls] += 1

    verb_class_wise_recall5 = []
    for cls in range(verb_num_classes):
        if verb_class_wise_frequency[cls] != 0:
           verb_class_wise_recall5.append(verb_class_wise_correct5[cls] / verb_class_wise_frequency[cls])
    noun_class_wise_recall5 = []
    for cls in range(noun_num_classes):
        if noun_class_wise_frequency[cls] != 0:
           noun_class_wise_recall5.append(noun_class_wise_correct5[cls] / noun_class_wise_frequency[cls])

    action_class_wise_recall5 = []
    for cls in range(action_num_classes):
        if action_class_wise_frequency[cls] != 0:
           action_class_wise_recall5.append(action_class_wise_correct5[cls] / action_class_wise_frequency[cls])

    verb_mean_recall5 = torch.Tensor(verb_class_wise_recall5).mean()*100
    noun_mean_recall5 = torch.Tensor(noun_class_wise_recall5).mean()*100
    action_mean_recall5 = torch.Tensor(action_class_wise_recall5).mean()*100

    print('[Verb Mean Recall @5: {:.2f}], [Noun Mean Recall @5: {:.2f}]'.format(verb_mean_recall5.item(), noun_mean_recall5.item()))
<<<<<<< HEAD
    print('[Action Mean Recall @5: {:.2f}]'.format(action_mean_recall5.item()))    
=======
    print('[Action Mean Recall @5: {:.2f}]'.format(action_mean_recall5.item()))
        
>>>>>>> e0ef9a0442f6ba31ffe45ac06f6b3bf13782c7de
    return None

def MAcc(test_score_dict, cfg):
    """
    Args:
        test_score_dict: Contains the prediciton scores of verbs and nouns along with the labels
    Returns:
        Mean Top-5 Recall
    """
    verb_num_classes = 19
    noun_num_classes = 51
    action_num_classes = 106
    verb_class_wise_frequency = np.zeros(verb_num_classes)
    verb_class_wise_correct = np.zeros(verb_num_classes)
    noun_class_wise_frequency = np.zeros(noun_num_classes)
    noun_class_wise_correct = np.zeros(noun_num_classes)
    action_class_wise_frequency = np.zeros(action_num_classes)
    action_class_wise_correct = np.zeros(action_num_classes)

    for nid in test_score_dict.keys():
        verb_scores = [ score_dict['verb'] for score_dict in test_score_dict[nid] ]
        noun_scores = [ score_dict['noun'] for score_dict in test_score_dict[nid] ]
        action_scores = [ score_dict['action'] for score_dict in test_score_dict[nid] ]

        verb_score = torch.mean(torch.stack(verb_scores), dim=0)
        noun_score = torch.mean(torch.stack(noun_scores), dim=0)
        action_score = torch.mean(torch.stack(action_scores), dim=0)
        verb_pred = torch.argmax(verb_score, dim=-1)
        noun_pred = torch.argmax(noun_score, dim=-1)
        action_pred = torch.argmax(action_score, dim=-1)

        verb_label = test_score_dict[nid][0]['verb_label']
        noun_label = test_score_dict[nid][0]['noun_label']
        action_label = test_score_dict[nid][0]['action_label']
        verb_correct = verb_label if verb_pred == verb_label else [] # [] if different and class_num if same
        noun_correct = noun_label if noun_pred == noun_label else [] # [] if different and class_num if same
        action_correct = action_label if action_pred == action_label else []# [] if different and class_num if same

        for cls in range(verb_num_classes):
            verb_class_wise_frequency[cls] += (verb_label == cls)
            verb_class_wise_correct[cls] += (verb_correct == cls)
        
        for cls in range(noun_num_classes):
            noun_class_wise_frequency[cls] += (noun_label == cls)
            noun_class_wise_correct[cls] += (noun_correct == cls)
        
        for cls in range(action_num_classes):
            action_class_wise_frequency[cls] += (action_label == cls)
            action_class_wise_correct[cls] += (action_correct == cls)
            
    verb_class_wise_recall = []
    for cls in range(verb_num_classes):
        if verb_class_wise_frequency[cls] != 0:
           verb_class_wise_recall.append(verb_class_wise_correct[cls] / verb_class_wise_frequency[cls])
    noun_class_wise_recall = []
    for cls in range(noun_num_classes):
        if noun_class_wise_frequency[cls] != 0:
           noun_class_wise_recall.append(noun_class_wise_correct[cls] / noun_class_wise_frequency[cls])
    action_class_wise_recall = []
    for cls in range(action_num_classes):
        if action_class_wise_frequency[cls] != 0:
           action_class_wise_recall.append(action_class_wise_correct[cls] / action_class_wise_frequency[cls])

    verb_mean_recall = torch.Tensor(verb_class_wise_recall).mean()*100
    noun_mean_recall = torch.Tensor(noun_class_wise_recall).mean()*100
    action_mean_recall = torch.Tensor(action_class_wise_recall).mean()*100

    print('[Verb Mean Acc : {:.2f}], [Noun Mean Acc: {:.2f}] [Action Mean Acc: {:.2f}]'\
        .format(verb_mean_recall.item(), noun_mean_recall.item(), action_mean_recall.item()))
        
    return None