import torch
import datasets
import csv
from tqdm import tqdm
from utils import Cubert_Model, LR_SP, NUM_WARMUP_STEPS, BATCH, CSV_FIELDS, EPOCHS
from utils import get_dataloader_input, get_dataloader
from utils import eval_fn, train_fn
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle


RESULTS_CSV_PATH = 'model_op/sp_results.csv'
MODEL_PATH = "model_op/model_sp_latest"
DEVICE = "cuda:4"


def train_cubert(train_data, dev_data):
    (model_input_ids, model_segment_ids,
        model_input_mask, model_labels_ids) = get_dataloader_input(train_data,
                                                                   example_types_to_evaluate="all",
                                                                   setting='ideal',
                                                                   vocab_file="pretrained_model_configs/vocab.txt")

    train_data_loader, train_file_length = get_dataloader(
        model_input_ids,
        model_input_mask,
        model_segment_ids,
        model_labels_ids,
        True
    )

    (model_input_ids, model_segment_ids,
        model_input_mask, model_labels_ids) = get_dataloader_input(dev_data,
                                                                   example_types_to_evaluate="all",
                                                                   setting='ideal',
                                                                   vocab_file="pretrained_model_configs/vocab.txt")

    dev_data_loader, dev_file_length = get_dataloader(
        model_input_ids,
        model_input_mask,
        model_segment_ids,
        model_labels_ids
    )

    device = torch.device(DEVICE)
    model = Cubert_Model(mode='train')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR_SP)
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
        train_fn(train_data_loader, model,
                 optimizer, device, scheduler)

        _, _, train_loss = eval_fn(
            train_data_loader, model, device)

        _, _, dev_loss = eval_fn(
            dev_data_loader, model, device)

        if(dev_loss < lowest_loss):
            torch.save(model.state_dict(), MODEL_PATH + '_best')
            lowest_loss = dev_loss

        # torch.save(model.state_dict(), MODEL_PATH + '_' + str(epoch))

        with open(RESULTS_CSV_PATH, "a") as f:
            results_row = [epoch, train_loss.item(), dev_loss.item()]

            csvwriter = csv.writer(f)
            csvwriter.writerow(results_row)


if __name__ == '__main__':
    # # evaluation
    # span_model = Cubert_Model()
    # span_model.to(DEVICE)
    # span_model.load_state_dict(torch.load("finetuned_ckpts/Cubert-1K", map_location=DEVICE))
    with open('resources/ideal_TRAIN.pkl', 'rb') as f:
        train_data = pickle.load(f)
    dev_data = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.VALIDATION)

    train_cubert(train_data, dev_data)
