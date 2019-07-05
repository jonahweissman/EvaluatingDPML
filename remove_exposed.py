from attack import load_data, attack_experiment
from classifier import get_predictions, train_private
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve

def attack_and_train_abridged_data(dataset, members_to_exclude=[],
                                   acceptable_fpr=0.01):
    """
    Trains a target model, then performs a Shokri attack to see who was
    exposed

    :param tuple dataset: train_x, train_y, test_x, test_y
    :param list members_to_exclude: indices to skip for training
    :param float acceptable_fpr: false positive rate that can tolerated
    :returns: dict containing exposed members, and training and testing
              accuracy
    """
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

    # train target model on abridged dataset
    classifier, _, _, _, train_acc, test_acc = train_private(
            (abridged_train_x, abridged_train_y, test_x, test_y),
            n_hidden=256,
            epochs=100,
            learning_rate=0.01,
            batch_size=200,
            model='softmax',
            l2_ratio=1e-7,
            silent=False,
            privacy='grad_pert',
            dp='rdp',
            epsilon=50.0,
            delta=1e-4)

    # predictions on data used in training (members)
    _, target_model_train_predictions = get_predictions(classifier.predict(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x={'x': abridged_train_x},
            num_epochs=1,
            shuffle=False)))
    
    # predictions on data not used in training
    _, target_model_test_predictions = get_predictions(classifier.predict(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x={'x': test_x},
            num_epochs=1,
            shuffle=False)))

    attack_test_x = np.vstack(
            [target_model_train_predictions, target_model_test_predictions]
            ).astype('float32')
    # labels for predictions -- members (1) then non-members (0)
    attack_test_y = np.concatenate(
            [np.ones(abridged_train_x.shape[0]), np.zeros(test_x.shape[0])]
                ).astype('int32')

    # output classes that are used in attack_test data
    test_classes = np.concatenate([abridged_train_y, test_y])
    
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

    _, attack_pred = attack_experiment(attack_test_x, attack_test_y,
            test_classes, args)
    revealed_members_list = revealed_members(attack_test_y,
                                             attack_pred[:,1],
                                             acceptable_fpr)
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
            'target model accuracy': {
                'train': train_acc,
                'test': test_acc
            }
           }

def revealed_members(membership, prediction, acceptable_fpr):
    """
    Determines which members are revealed by membership inference

    :param ndarray membership: binary membership status for each element in
                               training set
    :param ndarray prediction: predicted odds of membership determined by
                               attack method
    :param float acceptable_fpr: false positive rate that can tolerated
    :return: list of indices of members that have been revealed
    """
    fpr, tpr, thresholds = roc_curve(membership, prediction, pos_label=1)
    l = list(filter(lambda x: x < acceptable_fpr, fpr))
    if len(l) == 0:
        print("Error: low acceptable fpr")
        return None
    threshold = thresholds[len(l)-1]
    preds = list(map(lambda val: 1 if val >= threshold else 0, prediction))
    tp = [a*b for a,b in zip(preds,membership)]
    revealed = list(map(lambda i: i if tp[i] == 1 else None, range(len(tp))))
    return list(filter(lambda x: x != None, revealed))

if __name__=='__main__':
    dataset = load_data('target_data.npz')
    all_exposed_members = set()
    while True:
        result = attack_and_train_abridged_data(dataset,
                                                list(all_exposed_members))
        print(result)

        if len(all_exposed_members) > 0:
            print(
              (f"After removing {len(all_exposed_members)-last_num_exposed}"
               f" members ({len(all_exposed_members)} total) and retraining,"
               "there are now"
               f" {len(result['exposed members'])} members exposed.")
            )
            print(
               (f"Test accuracy changed from {last_test_accuracy}"
                f" to {result['target model accuracy']['test']}.")
            )

        last_num_exposed = len(all_exposed_members)
        all_exposed_members.update(result['exposed members'])
        last_test_accuracy = result['target model accuracy']['test']

        if len(result['exposed members']) == 0:
            print((f"After removing {len(all_exposed_members)} members,"
                    " no more members are exposed."))
            break
