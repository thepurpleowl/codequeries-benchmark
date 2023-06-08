import torch
import datasets
import csv
from tqdm import tqdm
from utils import Relevance_Classification_Model, LR_RP, NUM_WARMUP_STEPS, RELEVANCE_BATCH, CSV_FIELDS, EPOCHS
from utils import get_relevance_dataloader_input, get_relevance_dataloader, DEVICE
from utils import eval_fn_relevance_prediction, train_fn_relevance_prediction
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
import pickle


def train_relevance_subsample(train_data, dev_data):
    RESULTS_CSV_PATH = 'model_op/rel_subsample_results.csv'
    MODEL_PATH = "model_op/model_rel_subsample"
    pos_train_data = train_data.filter(lambda x: x['relevance_label'] == 1)
    neg_train_data = train_data.filter(lambda x: x['relevance_label'] == 0).shuffle(seed=42).select(range(3 * pos_train_data.shape[0]))
    sampled_train_data = datasets.concatenate_datasets([pos_train_data, neg_train_data]).shuffle(seed=42)

    (model_input_ids, model_segment_ids,
        model_input_mask, model_labels_ids) = get_relevance_dataloader_input(sampled_train_data,
                                                                             vocab_file="pretrained_model_configs/vocab.txt")

    train_data_loader, train_file_length = get_relevance_dataloader(
        model_input_ids,
        model_input_mask,
        model_segment_ids,
        model_labels_ids,
        True
    )

    (model_input_ids, model_segment_ids,
        model_input_mask, model_labels_ids) = get_relevance_dataloader_input(dev_data,
                                                                             vocab_file="pretrained_model_configs/vocab.txt")

    dev_data_loader, dev_file_length = get_relevance_dataloader(
        model_input_ids,
        model_input_mask,
        model_segment_ids,
        model_labels_ids
    )

    device = torch.device(DEVICE)
    model = Relevance_Classification_Model(mode='train')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR_RP)
    num_train_steps = int(
        (train_file_length / RELEVANCE_BATCH) * EPOCHS
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=num_train_steps
    )

    lowest_loss = float("inf")

    with open(RESULTS_CSV_PATH, "a") as f:
        csvwriter = csv.writer(f)
        rel_CSV_FIELDS = CSV_FIELDS + ['Train scores', 'Dev scores']
        csvwriter.writerow(rel_CSV_FIELDS)

    for epoch in tqdm(range(EPOCHS)):
        train_fn_relevance_prediction(train_data_loader, model,
                                      optimizer, device, scheduler)

        train_outputs, train_targets, train_loss = eval_fn_relevance_prediction(
            train_data_loader, model, device)
        train_scores = precision_recall_fscore_support(train_targets, train_outputs)

        dev_outputs, dev_targets, dev_loss = eval_fn_relevance_prediction(
            dev_data_loader, model, device)
        dev_scores = precision_recall_fscore_support(dev_targets, dev_outputs)

        if(dev_loss < lowest_loss):
            torch.save(model.state_dict(), MODEL_PATH + '_best')
            lowest_loss = dev_loss

        # torch.save(model.state_dict(), MODEL_PATH + '_' + str(epoch))

        with open(RESULTS_CSV_PATH, "a") as f:
            results_row = [epoch, train_loss.item(), dev_loss.item(), str(train_scores), str(dev_scores)]

            csvwriter = csv.writer(f)
            csvwriter.writerow(results_row)


def train_relevance_subsample_ada(train_data, dev_data):
    RESULTS_CSV_PATH = 'model_op/rel_subsample_results_ada.csv'
    MODEL_PATH = "model_op/model_rel_subsample_ada"
    pos_train_data = train_data.filter(lambda x: x['relevance_label'] == 1)
    neg_train_data = train_data.filter(lambda x: x['relevance_label'] == 0).shuffle(seed=42)
    total_negative = pos_train_data.shape[0] * 3
    current_neg_train_data = neg_train_data.select(range(total_negative))
    buffer_neg_train_data = neg_train_data.select(range(total_negative, neg_train_data.shape[0]))
    train_file_length = pos_train_data.shape[0] * 4

    (model_input_ids, model_segment_ids,
        model_input_mask, model_labels_ids) = get_relevance_dataloader_input(dev_data,
                                                                             vocab_file="pretrained_model_configs/vocab.txt")

    dev_data_loader, dev_file_length = get_relevance_dataloader(
        model_input_ids,
        model_input_mask,
        model_segment_ids,
        model_labels_ids
    )

    device = torch.device(DEVICE)
    model = Relevance_Classification_Model(mode='train')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR_RP)
    num_train_steps = int(
        (train_file_length / RELEVANCE_BATCH) * EPOCHS
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=num_train_steps
    )

    lowest_loss = float("inf")

    with open(RESULTS_CSV_PATH, "a") as f:
        csvwriter = csv.writer(f)
        rel_CSV_FIELDS = CSV_FIELDS + ['Train scores', 'Dev scores']
        csvwriter.writerow(rel_CSV_FIELDS)

    for epoch in tqdm(range(EPOCHS)):
        # adaptively select neg examples
        sampled_train_data = datasets.concatenate_datasets([pos_train_data, current_neg_train_data]).shuffle(seed=42)

        (model_input_ids, model_segment_ids,
            model_input_mask, model_labels_ids) = get_relevance_dataloader_input(sampled_train_data,
                                                                                 vocab_file="pretrained_model_configs/vocab.txt")

        train_data_loader, train_file_length = get_relevance_dataloader(
            model_input_ids,
            model_input_mask,
            model_segment_ids,
            model_labels_ids
        )

        train_fn_relevance_prediction(train_data_loader, model,
                                      optimizer, device, scheduler)

        train_outputs, train_targets, train_loss = eval_fn_relevance_prediction(
            train_data_loader, model, device)
        train_scores = precision_recall_fscore_support(train_targets, train_outputs)

        dev_outputs, dev_targets, dev_loss = eval_fn_relevance_prediction(
            dev_data_loader, model, device)
        dev_scores = precision_recall_fscore_support(dev_targets, dev_outputs)

        if(dev_loss < lowest_loss):
            torch.save(model.state_dict(), MODEL_PATH + '_best')
            lowest_loss = dev_loss

        # torch.save(model.state_dict(), MODEL_PATH + '_' + str(epoch))

        with open(RESULTS_CSV_PATH, "a") as f:
            results_row = [epoch, train_loss.item(), dev_loss.item(), str(train_scores), str(dev_scores)]

            csvwriter = csv.writer(f)
            csvwriter.writerow(results_row)

        # preprare negative data for next epoch
        retain_old = int(total_negative * 0.5)
        retrieve_new = total_negative - retain_old
        retained_neg = current_neg_train_data.shuffle(seed=42).select(range(retain_old))
        new_neg = buffer_neg_train_data.shuffle(seed=42).select(range(retrieve_new))
        current_neg_train_data = datasets.concatenate_datasets([retained_neg, new_neg])


if __name__ == '__main__':
    with open('resources/twostep_TRAIN.pkl', 'rb') as f:
        train_data = pickle.load(f)
    dev_data = datasets.load_dataset("thepurpleowl/codequeries", "twostep", split=datasets.Split.VALIDATION).shuffle(seed=42).select(list(range(1000)))

    train_relevance_subsample_ada(train_data, dev_data)
