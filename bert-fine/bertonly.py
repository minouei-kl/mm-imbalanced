import numpy as np
import time
import datetime
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import math
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
import json
import tarfile
import os
from sklearn.metrics import confusion_matrix
from utils import *
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)
batch_size = 32


def get_dataset(tar_path):

    # txts = []
    # targets = []

    # with open(tar_path.replace('tar', 'json'), "rb") as json_file:
    #     json_data = json.load(json_file)

    # with tarfile.open(tar_path, mode="r|*") as stream:
    #     for info in stream:
    #         if not info.isfile():
    #             continue
    #         file_path = info.name
    #         if file_path is None or 'png' not in file_path:
    #             continue
    #         filename = os.path.basename(file_path)
    #         if not filename in json_data.keys():
    #             continue
    #         # data = stream.extractfile(info).read()
    #         parent_dir_name = os.path.basename(
    #             os.path.dirname(file_path))
    #         target = int(parent_dir_name)
    #         targets.append(target)
    #         txt = json_data[filename]
    #         txts.append(txt)

    # input_ids = []
    # attention_masks = []

    # for txt in txts:
    #     encoded_dict = tokenizer.encode_plus(
    #         txt,                      # Sentence to encode.
    #         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    #         pad_to_max_length=True, truncation=True, max_length=512,
    #         return_attention_mask=True,   # Construct attn. masks.
    #         return_tensors='pt',     # Return pytorch tensors.
    #     )
    #     # Add the encoded sentence to the list.
    #     input_ids.append(encoded_dict['input_ids'])
    #     # And its attention mask (simply differentiates padding from non-padding).
    #     attention_masks.append(encoded_dict['attention_mask'])

    # input_ids = torch.cat(input_ids, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(targets)

    dn = 'train/' if 'train' in tar_path else 'test/'

    # torch.save(input_ids, dn+'input_ids.pt')
    # torch.save(attention_masks, dn+'attention_masks.pt')
    # torch.save(labels, dn+'labels.pt')
    input_ids = torch.load(dn+'input_ids.pt')
    attention_masks = torch.load(dn+'attention_masks.pt')
    labels = torch.load(dn+'labels.pt')

    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset


tar_path = "/home/minouei/Downloads/datasets/rvl/imbalanced/train.tar"
train_dataset = get_dataset(tar_path)
tar_path = "/home/minouei/Downloads/datasets/rvl/imbalanced/test.tar"
val_dataset = get_dataset(tar_path)

train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size,  # Trains with this batch size.
    num_workers=10)

validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=batch_size,  # Evaluate with this batch size.
    num_workers=10
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=16,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
# model = torch.load('bert_model')
model = model.to(device)

optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

epochs = 4

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=1000,  # Default value in run_glue.py
#                                             num_training_steps=total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, epochs, len(train_dataloader))
tf_writer = SummaryWriter(
    log_dir=os.path.join('root_log', 'bert5'))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        # if False:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the device using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        optimizer.zero_grad()
        output = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels)
        loss = output.loss
        total_train_loss += loss.item()

        logits = output.logits
        acc1, acc5 = accuracy(logits, b_labels, topk=(1, 5))
        losses.update(loss.item(), b_input_ids.size(0))
        top1.update(acc1[0], b_input_ids.size(0))

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    tf_writer.add_scalar('loss/train', losses.avg, epoch_i)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch_i)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Tracking variables
    all_preds = []
    all_targets = []
    total_eval_accuracy = 0
    best_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
        loss = output.loss
        total_eval_loss += loss.item()
        # Move logits and labels to CPU if we are using GPU
        logits = output.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        all_preds.extend(pred_flat)
        all_targets.extend(labels_flat)

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    if avg_val_accuracy > best_eval_accuracy:
        torch.save(model, 'bert_model2')
        best_eval_accuracy = avg_val_accuracy
    #print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    #print("  Validation took: {:}".format(validation_time))
    # Record all statistics from this epoch.
    report_results(
        all_targets, all_preds, 'logs', str(epoch_i))
    cf = confusion_matrix(all_targets, all_preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    out_cls_acc = '%s Class Accuracy: %s' % ('flag', (np.array2string(
        cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
    print(out_cls_acc)

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
print("")
print("Training complete!")
torch.save(model, 'bert_model4')

print("Total training took {:} (h:mm:ss)".format(
    format_time(time.time()-total_t0)))
# flag Class Accuracy: [0.942, 0.906, 0.979, 0.938, 0.859, 0.860, 0.883, 0.893, 0.842, 0.784, 0.763, 0.746,
#                       0.569, 0.630, 0.959, 0.568]
