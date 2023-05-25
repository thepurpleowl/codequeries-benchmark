import sys
import torch
import datasets
import csv
from tqdm import tqdm
from utils import Relevance_Classification_Model, DEVICE, MAX_LEN, LEARNING_RATE, NUM_WARMUP_STEPS, BATCH, CSV_FIELDS, EPOCHS
from utils import get_relevance_dataloader_input, get_relevance_dataloader
from utils import eval_fn_relevance_prediction, all_metrics_scores, train_fn_relevance_prediction
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle

RESULTS_CSV_PATH = 'rel_results.csv'
MODEL_PATH = "model_rel_best"

def train_cubert(train_data, dev_data):
    (model_input_ids, model_segment_ids,
        model_input_mask, model_labels_ids) = get_relevance_dataloader_input(train_data,
                                                                vocab_file="pretrained_model_configs/vocab.txt")

    train_data_loader, train_file_length = get_relevance_dataloader(
        model_input_ids,
        model_input_mask,
        model_segment_ids,
        model_labels_ids
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
    model = Relevance_Classification_Model()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_train_steps = int(
        (train_file_length / BATCH) * EPOCHS
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=num_train_steps
    )

    lowest_loss = float("inf")

    with open(RESULTS_CSV_PATH, "a") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(CSV_FIELDS)

    for epoch in tqdm(range(EPOCHS)):
        train_fn_relevance_prediction(train_data_loader, model,
                        optimizer, device, scheduler)

        _, _, train_loss = eval_fn_relevance_prediction(
            train_data_loader, model, device)

        _, _, dev_loss = eval_fn_relevance_prediction(
            dev_data_loader, model, device)

        if(dev_loss < lowest_loss):
            torch.save(model.state_dict(), MODEL_PATH)
            lowest_loss = dev_loss

        with open(RESULTS_CSV_PATH, "a") as f:
            results_row = [epoch, train_loss.item(), dev_loss.item()]

            csvwriter = csv.writer(f)
            csvwriter.writerow(results_row)

if __name__ == '__main__':
    train_data = datasets.load_dataset("thepurpleowl/codequeries", "twostep", split=datasets.Split.TRAIN).select(list(range(200)))
    dev_data = datasets.load_dataset("thepurpleowl/codequeries", "twostep", split=datasets.Split.VALIDATION).select(list(range(100)))

    train_cubert(train_data, dev_data)