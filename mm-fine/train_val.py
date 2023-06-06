import torch
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from utils import *


def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, device='cpu', param_dict_model=None, param_dict_opt=None, save_best=False, file_path='./saved_models/best_model.pt', writer=None):
    """Train the BertClassifier model.
    """
    # Start training loop
    best_acc_val = 0
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            img_ip, text_ip, label = batch["image"], batch["BERT_ip"], batch['label']

            b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)

            imgs_ip = img_ip.to(device)

            b_labels = label.to(device)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(text=[b_input_ids, b_attn_mask],
                           image=imgs_ip)

            # print(logits)
            # print(b_labels)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Write onto tensorboard
                if writer != None:
                    writer.add_scalar(
                        'Training Loss', (batch_loss / batch_counts), epoch_i*len(train_dataloader)+step)

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(
                model, loss_fn, val_dataloader, device, epoch_i)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)

            # Write onto tensorboard
            if writer != None:
                writer.add_scalar('Validation Loss', val_loss, epoch_i+1)
                writer.add_scalar('Validation Accuracy',
                                  val_accuracy, epoch_i+1)

            # Save the best model
            if save_best:
                if val_accuracy > best_acc_val:
                    best_acc_val = val_accuracy
                    torch.save({
                        'epoch': epoch_i+1,
                        'model_params': param_dict_model,
                        'opt_params': param_dict_opt,
                        'model_state_dict': model.state_dict(),
                        'opt_state_dict': optimizer.state_dict(),
                        'sch_state_dict': scheduler.state_dict()
                    }, file_path)

        print("\n")

    torch.save({
        'epoch': epoch_i+1,
        'model_params': param_dict_model,
        'opt_params': param_dict_opt,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': optimizer.state_dict(),
        'sch_state_dict': scheduler.state_dict()
    }, './saved_models/lastepoch.pt')
    print("Training complete!")


def evaluate(model, loss_fn, val_dataloader, device, epoch_i):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    all_preds = []
    all_targets = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        img_ip, text_ip, label = batch["image"], batch["BERT_ip"], batch['label']

        b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)

        imgs_ip = img_ip.to(device)

        b_labels = label.to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(text=[b_input_ids, b_attn_mask],
                           image=imgs_ip, label=b_labels)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(b_labels.cpu().numpy())

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    report_results(
        all_targets, all_preds, 'logs', str(epoch_i))
    cf = confusion_matrix(all_targets, all_preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    out_cls_acc = '%s Class Accuracy: %s' % ('flag', (np.array2string(
        cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
    print(out_cls_acc)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy
