from attack import load_data, attack_experiment, membership_inference, train_target_model
import numpy as np
from sklearn.metrics import roc_curve

def attack_and_train_abridged_data(dataset, attack, members_to_exclude=[]):
    """
    Trains a target model, then performs a Shokri attack to see who was
    exposed

    :param tuple dataset: train_x, train_y, test_x, test_y
    :param str attack: use Shokri or Yeom attack to determine exposure
    :param list members_to_exclude: indices to skip for training
    :param float acceptable_fpr: false positive rate that can tolerated
    :returns: dict containing exposed members, and training and testing
              accuracy
    """
    assert attack in ['Yeom', 'Shokri'], "Invalid attack type"
    train_x, train_y, test_x, test_y = dataset
    # `mask` has the same length as the training data and is True for rows
    # we want to keep in `abridged_dataset` and False for rows we will ignore
    mask = np.ones((len(train_y),), dtype=bool)
    mask[members_to_exclude] = False
    abridged_train_x = train_x[mask]
    abridged_train_y = train_y[mask] 
    # we won't be using the unabridged versions anymore
    del train_x
    del train_y

    attack_test_x, attack_test_y, test_classes, train_loss, _, train_acc, test_acc \
    = train_target_model(
            (abridged_train_x, abridged_train_y, test_x, test_y),
            n_hidden=256,
            epochs=100,
            learning_rate=0.01,
            batch_size=200,
            model='softmax',
            l2_ratio=1e-7,
            save=False,
            privacy='no_privacy')

    if attack == 'Shokri':
        class args:
            target_epochs = 100
            target_batch_size = 200
            target_learning_rate = 0.01
            n_shadow = 5
            target_n_hidden = 256
            target_l2_ratio = 1e-7
            target_model = 'softmax'
            save_model = False
            attack_epochs = 100
            attack_batch_size = 100
            attack_learning_rate = 0.01
            attack_n_hidden = 64
            attack_l2_ratio = 1e-6
            attack_model = 'nn'

        _, attack_pred = attack_experiment(args, attack_test_x, attack_test_y,
                test_classes)
        revealed_members_list = revealed_members(
                attack_test_y,
                attack_pred,
                threshold=0.5)
        n_fp = len(false_positives(attack_test_y, classify(attack_pred, 0.5)))
    elif attack == 'Yeom':
        train_loss *= 0.1
        _, mem_pred = membership_inference(
                np.append(abridged_train_y, test_y),
                attack_test_x,
                attack_test_y,
                train_loss)
        revealed_members_list = revealed_members(
                attack_test_y,
                mem_pred,
                threshold=train_loss,
                below_threshold_positive=True)
        n_fp = len(false_positives(attack_test_y,
            classify(mem_pred, train_loss, below_threshold_positive=True)))
    n_tp = len(revealed_members_list)
    if n_tp + n_fp > 0:
        ppv = n_tp / (n_tp + n_fp)
    else:
        ppv = 0.5
    # adjust for missing indices:
    # go through each removed index and 'reinsert' it,
    # starting with the lowest index and working up
    for mem_excluded_idx in sorted(members_to_exclude):
        for i, revealed_mem_idx in enumerate(revealed_members_list):
            if mem_excluded_idx <= revealed_mem_idx:
                # the index of this revealed member would have been 1 higher
                # if the current excluded member had been included
                revealed_members_list[i] += 1
    return {
            'exposed members': revealed_members_list,
            'positive predictive value': ppv,
            'target model accuracy': {
                'train': train_acc,
                'test': test_acc
            }
           }

def revealed_members(membership, prediction, threshold=None, acceptable_fpr=None,
        below_threshold_positive=False):
    """
    Determines which members are revealed by membership inference
    
    :param ndarray membership: binary membership status for each element in population
    :param ndarray prediction: chance of membership determined by attack method (not necessarily 0-1)
    :param float threshold: predictions above threshold will be classified as members
    :param float acceptable_fpr: false positive rate that can tolerated
                                 (optional way of setting threshold)
    :param below_threshold_positive: see classify doc string
    :return: list of indices of members that have been revealed
    """
    if threshold is None:
        assert acceptable_fpr is not None, "Must specify either `threshold` or `acceptable_fpr`"
        threshold = threshold_fixed_fpr(membership, prediction, acceptable_fpr)
    return true_positives(membership,
            classify(prediction, threshold, below_threshold_positive=below_threshold_positive))

def threshold_fixed_fpr(membership, prediction, acceptable_fpr):
    """Returns the lowest prediction threshold that yields FPR < acceptable_fpr"""
    fpr, tpr, thresholds = roc_curve(membership, prediction, pos_label=1)
    l = list(filter(lambda x: x < acceptable_fpr, fpr))
    if len(l) == 0:
        print("Error: low acceptable fpr")
        return None
    return thresholds[len(l)-1]

def classify(prediction, threshold, below_threshold_positive):
    """classify predictions/weights by less than threshold (False) or greater (True)
       (reversed if below_threshold_positive is true"""
    if below_threshold_positive:
        return list(map(lambda val: val <= threshold, prediction))
    else:
        return list(map(lambda val: val >= threshold, prediction))

def true_positives(membership, predicted_membership):
    """List of indices that are correctly predicted as members"""
    tp_status = [m and p_m for m, p_m in zip(membership, predicted_membership)]
    return list(np.where(np.array(tp_status))[0])

def false_positives(membership, predicted_membership):
    """List of indices that are incorrectly predicted as members"""
    fp_status = [p_m and not m for m, p_m in zip(membership, predicted_membership)]
    return list(np.where(np.array(fp_status))[0])

if __name__=='__main__':
    dataset = load_data('target_data.npz')
    all_exposed_members = set()
    with open('exposed_members.txt', 'r') as f:
        all_exposed_members = set([int(m) for m in f.readlines()])
    runs = 0

    while True:
        result = attack_and_train_abridged_data(
                dataset,
                'Yeom',
                members_to_exclude=list(all_exposed_members))
        print(f"Run {runs}")
        print(result)
        with open('exposed_members.txt', 'a') as f:
            for exposed_member in result['exposed members']:
                print(exposed_member, file=f)

        if runs > 0:
            print(
              (f"After removing {len(all_exposed_members)-last_num_exposed}"
               f" members ({len(all_exposed_members)} total) and retraining,"
               f" there are now"
               f" {len(result['exposed members'])} members exposed."
               f" (PPV: {result['positive predictive value']:.2%})")
              )
            print(
               (f"Test accuracy changed from {last_test_accuracy}"
                f" to {result['target model accuracy']['test']}."))

        last_num_exposed = len(all_exposed_members)
        all_exposed_members.update(result['exposed members'])
        last_test_accuracy = result['target model accuracy']['test']
        runs += 1
