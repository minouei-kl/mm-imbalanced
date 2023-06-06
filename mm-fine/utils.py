import numpy as np
import torch
import math
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

CLASSES = ["letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication",
           "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]


def get_cosine_schedule_with_warmup(optimizer, epochs, n_steps):
    def get_cosine_schedule_with_warmup(
        optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
    ):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / \
                float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch, verbose=False)

    warmup_proportion = 1/epochs
    # n_steps = int(np.ceil(n_samples / batch_size))
    num_training_steps = n_steps * epochs
    num_warmup_steps = int(warmup_proportion * num_training_steps)
    sch = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps)
    return sch


def report_results(y_true, y_pred, store_name, epoch=''):
    report = classification_report(y_true, y_pred, output_dict=True)
    print(json.dumps(report, indent=4))
    matrix = confusion_matrix(y_true, y_pred)
    class_acc = {'class_acc': matrix.diagonal() /
                 matrix.sum(axis=1).tolist()}
    r_path = os.path.join(store_name, epoch + 'result.txt')
    with open(r_path, 'w') as fp:
        fp.write(str({**report, **class_acc}))
        fp.write(str(class_acc))

    my_plot_confusion_matrix(y_true, y_pred, CLASSES, True)
    plt.savefig(os.path.join(store_name, epoch + 'confusion_matrix.png'))


def my_plot_confusion_matrix(y_true, y_pred, classes,
                             normalize=False,
                             title=None,
                             cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('confusion_matrix.png', format='png')
    return ax
