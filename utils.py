import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import math
from transformers import BertConfig
from transformers import BertModel
from collections import namedtuple


NUMBER_OF_LABELS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 1024
MAX_LEN_RELEVANCE = 512
SECOND_LAYER_HIDDEN_SIZE = 2048
BATCH = 4
RELEVANCE_BATCH = 16
DROPOUT = 0.1
LEARNING_RATE = 3e-5
EPOCHS = 3
NUM_WARMUP_STEPS = 0

CSV_FIELDS = ["Epoch", "Train_Loss", "Validation_Loss"]
USE_WEIGHTS_IN_LOSS_FUNCTION = False
RESULTS_FILE_PATH = "training_results_.csv"
MODEL_STORE_PATH = "trained_model_"


class PREPAREVOCAB:
    def __init__(self, vocab_file: str):
        self.vocab_file = vocab_file

    def load_vocab(self):
        """
        This function created a Cubert vocabulary to ids dictionary.
        """
        vocab = {}
        with open(self.vocab_file, 'r') as f:
            for j, line in enumerate(f):
                line = line.strip()
                vocab[line] = j
        return vocab

    def convert_by_vocab(self, items: list) -> list:
        """
        This function takes a list of subtokens and converts them to ids according
        to Cubert vocabulary.
        """
        vocab = self.load_vocab()
        output = []
        for item in items:
            item = "'" + item + "'"
            try:
                output.append(vocab[item])
            except KeyError:
                output.append(vocab["'[UNK]_'"])

        return output


# dataset
class JSONSpanDataset:
    def __init__(self, all_input_ids, all_input_mask,
                 all_segment_ids, all_labels_ids):
        self.all_input_ids = all_input_ids
        self.all_input_mask = all_input_mask
        self.all_segment_ids = all_segment_ids
        self.all_labels_ids = all_labels_ids
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, item):
        input_ids = self.all_input_ids[item]
        segment_ids = self.all_segment_ids[item]
        input_mask = self.all_input_mask[item]
        labels_ids = self.all_labels_ids[item]

        if(len(input_ids) > self.max_len):
            input_ids = input_ids[:self.max_len]
            segment_ids = segment_ids[:self.max_len]
            input_mask = input_mask[:self.max_len]
            labels_ids = labels_ids[:self.max_len]

        elif(len(input_ids) < self.max_len):
            while(len(input_ids) < self.max_len):
                input_ids.append(0)
                segment_ids.append(0)
                input_mask.append(0)
                labels_ids.append(4)

        assert len(input_ids) == self.max_len
        assert len(input_mask) == self.max_len
        assert len(segment_ids) == self.max_len
        assert len(labels_ids) == self.max_len

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(input_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(segment_ids, dtype=torch.long),
            "targets": torch.tensor(labels_ids, dtype=torch.long),
        }


class JSONRelevanceDataset:
    def __init__(self, all_input_ids, all_input_mask,
                 all_segment_ids, all_relevance_labels):
        self.all_input_ids = all_input_ids
        self.all_input_mask = all_input_mask
        self.all_segment_ids = all_segment_ids
        self.all_relevance_labels = all_relevance_labels
        self.max_len = MAX_LEN_RELEVANCE

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, item):
        input_ids = self.all_input_ids[item]
        segment_ids = self.all_segment_ids[item]
        input_mask = self.all_input_mask[item]
        relevance_label = self.all_relevance_labels[item]

        if(len(input_ids) > self.max_len):
            while(len(input_ids) > self.max_len):
                input_ids.pop()
                segment_ids.pop()
                input_mask.pop()

        elif(len(input_ids) < self.max_len):
            while(len(input_ids) < self.max_len):
                input_ids.append(0)
                segment_ids.append(0)
                input_mask.append(0)

        assert len(input_ids) == self.max_len
        assert len(input_mask) == self.max_len
        assert len(segment_ids) == self.max_len

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(input_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(segment_ids, dtype=torch.long),
            "targets": torch.tensor(relevance_label, dtype=torch.long),
        }


# model
class Cubert_Model(nn.Module):
    def __init__(self, config_path='./pretrained_model_configs/config_1024.json'):
        super().__init__()
        config = BertConfig.from_pretrained(config_path)
        self.bert = BertModel(config)
        self.bert_drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(MAX_LEN, NUMBER_OF_LABELS)

    def forward(self, ids, mask, token_type_ids):
        o = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o = o[0]
        o = self.bert_drop(o)
        output = self.out(o)
        return output


class Relevance_Classification_Model(nn.Module):
    def __init__(self, config_path='./pretrained_model_configs/config_512.json'):
        super().__init__()
        config = BertConfig.from_pretrained(config_path)
        self.bert = BertModel(config)
        self.bert_drop = nn.Dropout(DROPOUT)
        self.out1 = nn.Linear(1024, SECOND_LAYER_HIDDEN_SIZE)
        self.out = nn.Linear(SECOND_LAYER_HIDDEN_SIZE, 2)
        self.relu = nn.ReLU()

    def forward(self, ids, mask, token_type_ids):
        o = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o = o[1]
        o = self.bert_drop(o)
        output = self.out1(o)
        output = self.relu(output)
        output = self.out(output)
        return output


# evaluation
def loss_fn(outputs, targets, batch, device):
    return nn.CrossEntropyLoss()(
        outputs.view(MAX_LEN * batch, NUMBER_OF_LABELS),
        targets.view(MAX_LEN * batch))


def eval_fn(data_loader, model, device):
    model.eval()
    accessing_target_sequence_for_first_time = 1
    accessing_output_sequence_for_first_time = 1
    total_loss = 0
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device)
            token_type_ids = token_type_ids.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

            loss = loss_fn(outputs, targets, outputs.shape[0], device)
            total_loss = total_loss + loss

            a = targets.data
            a = a.view(a.shape[0] * a.shape[1])

            if(accessing_target_sequence_for_first_time):
                target_sequences = np.array(targets.cpu().detach().numpy())
                accessing_target_sequence_for_first_time = 0
            else:
                target_sequences = np.concatenate((target_sequences,
                                                   targets.cpu().detach().numpy()), axis=0)

            b = outputs.data
            out = torch.max(b.clone().detach(), 2)[1]

            if(accessing_output_sequence_for_first_time):
                output_sequences = np.array(out.cpu().detach().numpy())
                accessing_output_sequence_for_first_time = 0
            else:
                output_sequences = np.concatenate((output_sequences,
                                                   out.cpu().detach().numpy()), axis=0)

            b = out.view(out.shape[0] * out.shape[1])

    return target_sequences, output_sequences, total_loss


def eval_fn_relevance_prediction(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader), desc="Relevance evaluation"):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device)
            token_type_ids = token_type_ids.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

            a = targets.data
            b = outputs.data                        # shape[batch, labels]
            b = torch.max(b.clone().detach(), 1)[1]

            temp_b = []
            for i in range(len(b)):
                temp_b.append(b[i].item())

            temp_a = []
            for i in range(len(a)):
                temp_a.append(a[i].item())

            fin_outputs.extend(temp_b)
            fin_targets.extend(temp_a)

    return fin_outputs, fin_targets


# training
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for d in tqdm(data_loader, total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device)
        token_type_ids = token_type_ids.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(outputs, targets, outputs.shape[0], device)
        loss.backward()
        optimizer.step()
        scheduler.step()


# utilities
def get_first_sep_label_index(sequence):
    for i, x in enumerate(sequence[:-1]):
        if((sequence[i + 1] != 4) and (x == 4)):
            return i


def should_add_to_eval_set(example_instance, example_type):
    if(example_type == "positive" and example_instance['example_type'] != 1):
        return False
    elif(example_type == "negative" and example_instance['example_type'] != 0):
        return False
    if(example_type == "positive" and example_instance['example_type'] == 1):
        return True
    elif(example_type == "negative" and example_instance['example_type'] == 0):
        return True
    elif(example_type == "all"):
        return True


def get_dataloader_input(examples_data, example_types_to_evaluate, setting, vocab_file):
    prepare_vocab_object = PREPAREVOCAB(vocab_file)
    sep_token = '[SEP]_'
    sep_token_id = prepare_vocab_object.convert_by_vocab([sep_token])[0]
    model_input_ids = []
    model_segment_ids = []
    model_input_mask = []
    model_labels_ids = []
    for example_instance in tqdm(examples_data, desc="Preparing input"):
        assert (len(example_instance['subtokenized_input_sequence'])
                == len(example_instance['label_sequence']))
        if(not should_add_to_eval_set(example_instance,
                                      example_types_to_evaluate)):
            continue

        instance_input_ids = []
        instance_segment_ids = []
        instance_input_mask = []
        instance_labels_ids = []
        if setting == "prefix":
            subtokens_ids = example_instance['subtokenized_input_sequence']
        else:
            subtokens_ids = prepare_vocab_object.convert_by_vocab(
                example_instance['subtokenized_input_sequence'])
        subtoken_labels = example_instance['label_sequence']
        segment_id = 0
        input_mask = 1
        for st_id, st_label in zip(subtokens_ids, subtoken_labels):
            instance_input_ids.append(st_id)
            instance_segment_ids.append(segment_id)
            instance_input_mask.append(input_mask)
            instance_labels_ids.append(st_label)

            if(st_id == sep_token_id
                    and segment_id == 0):
                # change segment id on first sep
                segment_id = 1
        assert segment_id == 1

        assert len(instance_input_ids) == len(instance_labels_ids) == len(
            instance_input_mask) == len(instance_segment_ids)
        model_input_ids.append(instance_input_ids)
        model_input_mask.append(instance_input_mask)
        model_segment_ids.append(instance_segment_ids)
        model_labels_ids.append(instance_labels_ids)

    return model_input_ids, model_segment_ids, model_input_mask, model_labels_ids


def prepare_sliding_window_input(model_input_ids, model_segment_ids, model_input_mask, model_labels_ids, sliding_window_width):
    formatted_model_input_ids = []
    formatted_model_segment_ids = []
    formatted_model_input_mask = []
    formatted_model_labels_ids = []

    example_id = 0
    for i in tqdm(range(len(model_input_ids)), desc="Prepare sliding window examples"):
        assert (len(model_input_ids[i])
        == len(model_segment_ids[i])
        == len(model_input_mask[i])
        == len(model_labels_ids[i]))

        example_len = len(model_input_ids[i])
        if example_len < sliding_window_width:
            formatted_model_input_ids.append(model_input_ids[i])
            formatted_model_segment_ids.append(model_segment_ids[i])
            formatted_model_input_mask.append(model_input_mask[i])
            formatted_model_labels_ids.append((example_id, 0, model_labels_ids[i]))
        else:
            first_sep_index = get_first_sep_label_index(model_labels_ids[i])

            query_subtoken_prefix_len = first_sep_index + 1
            single_code_block_length = sliding_window_width - query_subtoken_prefix_len

            query_subtoken_prefix = model_input_ids[i][:query_subtoken_prefix_len]
            split_prefix_segment_ids = model_segment_ids[i][:query_subtoken_prefix_len]
            split_prefix_input_mask = model_input_mask[i][:query_subtoken_prefix_len]
            split_prefix_labels_ids = model_labels_ids[i][:query_subtoken_prefix_len]
            for k in range(len(query_subtoken_prefix)):
                assert split_prefix_segment_ids[k] == 0
                assert split_prefix_input_mask[k] == 1
                assert split_prefix_labels_ids[k] == 4

            split = 0
            current_index = first_sep_index + 1
            while(current_index < example_len):
                sw_start = current_index
                sw_end = min(current_index + single_code_block_length, example_len)

                split_model_input_ids = query_subtoken_prefix.copy()
                split_model_segment_ids = split_prefix_segment_ids.copy()
                split_model_input_mask = split_prefix_input_mask.copy()
                split_model_labels_ids = split_prefix_labels_ids.copy()
                split_model_input_ids.extend(model_input_ids[i][sw_start:sw_end])
                split_model_segment_ids.extend(model_segment_ids[i][sw_start:sw_end])
                split_model_input_mask.extend(model_input_mask[i][sw_start:sw_end])
                split_model_labels_ids.extend(model_labels_ids[i][sw_start:sw_end])

                formatted_model_input_ids.append(split_model_input_ids)
                formatted_model_segment_ids.append(split_model_segment_ids)
                formatted_model_input_mask.append(split_model_input_mask)
                formatted_model_labels_ids.append((example_id, split, split_model_labels_ids))
                # increment split
                split += 1
                current_index += single_code_block_length
        # increment example id
        example_id += 1

    return formatted_model_input_ids, formatted_model_segment_ids, formatted_model_input_mask, formatted_model_labels_ids


def prepare_sliding_window_output(target_sequences, pruned_target_sequences, output_sequences, sliding_window_width):
    aggregated_target_sequences = []
    aggregated_pruned_target_sequences = []
    aggregated_output_sequences = []
    assert len(target_sequences) == len(pruned_target_sequences) == len(output_sequences)

    k = 0
    while k < len(target_sequences):
        current_target_sequence = target_sequences[k][2]
        current_pruned_target_sequence = pruned_target_sequences[k]
        current_output_sequence = output_sequences[k]

        current_example_id = target_sequences[k][0]
        current_split_id = target_sequences[k][1]
        assert current_split_id == 0
        k_end = k + 1

        first_sep_index = get_first_sep_label_index(current_target_sequence)
        assert get_first_sep_label_index(current_pruned_target_sequence) == first_sep_index
        query_subtoken_prefix_len = first_sep_index + 1

        while(k_end < len(target_sequences)
                and target_sequences[k_end][0] == current_example_id):
            # check split increment
            assert target_sequences[k_end][1] == current_split_id + 1
            # check the query length
            assert get_first_sep_label_index(target_sequences[k_end][2]) == first_sep_index
            assert get_first_sep_label_index(pruned_target_sequences[k_end]) == first_sep_index

            current_target_sequence.extend(target_sequences[k_end][2][query_subtoken_prefix_len:])
            current_pruned_target_sequence.extend(pruned_target_sequences[k_end][query_subtoken_prefix_len:])
            current_output_sequence.extend(output_sequences[k_end][query_subtoken_prefix_len:])

            current_split_id += 1
            k_end += 1
        # add the aggregated example
        assert len(current_pruned_target_sequence) == len(current_output_sequence)
        number_of_splits = k_end - k
        assert number_of_splits == (math.ceil((len(current_target_sequence) - query_subtoken_prefix_len)
                                    / (sliding_window_width - query_subtoken_prefix_len)))
        assert (len(current_output_sequence)
                == (number_of_splits * sliding_window_width - (number_of_splits - 1) * query_subtoken_prefix_len))
        aggregated_target_sequences.append(current_target_sequence)
        aggregated_pruned_target_sequences.append(current_pruned_target_sequence)
        aggregated_output_sequences.append(current_output_sequence)

        k = k_end

    return aggregated_target_sequences, aggregated_pruned_target_sequences, aggregated_output_sequences


def get_examples_for_twostep(examples_data, example_types_to_evaluate):
    twostep_dict = {}
    for i, example_instance in enumerate(examples_data):
        if(not should_add_to_eval_set(example_instance,
                                      example_types_to_evaluate)):
            continue

        twostep_key = (example_instance['query_name'], example_instance['code_file_path'])
        if twostep_key not in twostep_dict:
            twostep_dict[twostep_key] = [i]
        else:
            twostep_dict[twostep_key].append(i)

    return twostep_dict


def get_twostep_dataloader_input(examples_data, example_types_to_evaluate, vocab_file, relevance_model):
    twostep_dict = get_examples_for_twostep(examples_data, example_types_to_evaluate)

    prepare_vocab_object = PREPAREVOCAB(vocab_file)
    sep_token = '[SEP]_'
    sep_token_id = prepare_vocab_object.convert_by_vocab([sep_token])[0]
    model_input_ids = []
    model_segment_ids = []
    model_input_mask = []
    model_labels_ids = []
    model_label_metadata_ids = []
    model_target_labels_ids = []
    for twostep_key in tqdm(twostep_dict.keys(), desc="Preparing twostep data"):
        # get relevant predicted examples
        first_sep_index = get_first_sep_label_index(examples_data[twostep_dict[twostep_key][0]]['label_sequence'])
        assert examples_data[twostep_dict[twostep_key][0]]['subtokenized_input_sequence'][first_sep_index] == sep_token
        # initiate span prediction examples
        instance_input_tokens = examples_data[twostep_dict[twostep_key][0]]['subtokenized_input_sequence'][:first_sep_index+1]
        instance_segment_ids = [0 for _ in range(first_sep_index+1)]
        instance_input_mask = [1 for _ in range(first_sep_index+1)]
        instance_target_labels_ids = [(x, -1, -1)
                                      for x in examples_data[twostep_dict[twostep_key][0]]['label_sequence'][:first_sep_index+1]]
        instance_predicted_labels_ids = instance_target_labels_ids.copy()
        # get relevance prediction of blocks
        if len(twostep_dict[twostep_key]) == 1:
            rel_i_out = [1]
            rel_i_targets = [1]
        else:
            # get key specific examples data
            twostep_key_relevance_data = []
            for ti in twostep_dict[twostep_key]:
                twostep_key_relevance_data.append(examples_data[ti])
                assert get_first_sep_label_index(examples_data[ti]['label_sequence']) == first_sep_index

            (rel_i_input_ids, rel_i_segment_ids,
            rel_i_input_mask, rel_i_labels_ids) = get_relevance_dataloader_input(twostep_key_relevance_data, vocab_file)
            rel_i_data_loader, _ = get_relevance_dataloader(
                rel_i_input_ids,
                rel_i_input_mask,
                rel_i_segment_ids,
                rel_i_labels_ids
            )
            rel_i_out, rel_i_targets = eval_fn_relevance_prediction(rel_i_data_loader, relevance_model, DEVICE)

        assert len(rel_i_out) == len(twostep_dict[twostep_key]) == len(rel_i_targets)
        # form span prediction examples
        extra_sep_added = False
        target_extra_sep_added = False
        for k, predicted in enumerate(rel_i_out):
            assert get_first_sep_label_index(examples_data[twostep_dict[twostep_key][k]]['label_sequence']) == first_sep_index
            block_tokens = examples_data[twostep_dict[twostep_key][k]]['subtokenized_input_sequence'][first_sep_index+1:]
            block_labels = examples_data[twostep_dict[twostep_key][k]]['label_sequence'][first_sep_index+1:]
            assert len(block_tokens) == len(block_labels)

            block_index = examples_data[twostep_dict[twostep_key][k]]['context_block']['index']
            if predicted == 1:
                block_length = len(block_tokens)

                instance_input_tokens.extend(block_tokens)
                instance_segment_ids.extend([1 for _ in range(block_length)])
                instance_input_mask.extend([1 for _ in range(block_length)])
                block_offset = 0
                for x in block_labels:
                    instance_predicted_labels_ids.append((x, block_index, block_offset))
                    block_offset += 1
                
                instance_input_tokens.append(sep_token)
                instance_segment_ids.append(1)
                instance_input_mask.append(1)
                instance_predicted_labels_ids.append((4, -1, -1))
                extra_sep_added = True

            if rel_i_targets[k] == 1:
                block_offset = 0
                for x in examples_data[twostep_dict[twostep_key][k]]['label_sequence'][first_sep_index+1:]:
                    instance_target_labels_ids.append((x, block_index, block_offset))
                    block_offset += 1

                instance_target_labels_ids.append((4, -1, -1))
                target_extra_sep_added = True

        # remove extra sep
        if extra_sep_added:
            instance_input_tokens.pop()
            instance_segment_ids.pop()
            instance_input_mask.pop()
            instance_predicted_labels_ids.pop()
        if target_extra_sep_added:
            instance_target_labels_ids.pop()

        instance_input_ids = prepare_vocab_object.convert_by_vocab(instance_input_tokens)
        instance_labels_ids = [y[0] for y in instance_predicted_labels_ids]

        assert len(instance_input_ids) == len(instance_labels_ids) == len(instance_input_mask) == len(instance_segment_ids)
        model_input_ids.append(instance_input_ids)
        model_input_mask.append(instance_input_mask)
        model_segment_ids.append(instance_segment_ids)
        model_labels_ids.append(instance_labels_ids)

        model_label_metadata_ids.append(instance_predicted_labels_ids)
        model_target_labels_ids.append(instance_target_labels_ids)

    return (model_input_ids, model_segment_ids, model_input_mask, model_labels_ids,
            model_label_metadata_ids, model_target_labels_ids)


def get_relevance_dataloader_input(examples_data, vocab_file, disable_tqdm=True):
    prepare_vocab_object = PREPAREVOCAB(vocab_file)
    sep_token = '[SEP]_'
    sep_token_id = prepare_vocab_object.convert_by_vocab([sep_token])[0]
    model_input_ids = []
    model_segment_ids = []
    model_input_mask = []
    model_labels_ids = []
    for i, example_instance in enumerate(tqdm(examples_data, desc="Preparing relevance data", disable=disable_tqdm)):
        instance_input_ids = []
        instance_segment_ids = []
        instance_input_mask = []
        instance_labels_ids = []
        subtokens_ids = prepare_vocab_object.convert_by_vocab(
            example_instance['subtokenized_input_sequence'])

        segment_id = 0
        input_mask = 1
        for st_id in subtokens_ids:
            instance_input_ids.append(st_id)
            instance_segment_ids.append(segment_id)
            instance_input_mask.append(input_mask)

            if(st_id == sep_token_id
                    and segment_id == 0):
                # change segment id on first sep
                segment_id = 1
        assert segment_id == 1
        instance_labels_ids.append(example_instance['relevance_label'])

        assert len(instance_input_ids) == len(instance_input_mask) == len(instance_segment_ids)

        model_input_ids.append(instance_input_ids)
        model_input_mask.append(instance_input_mask)
        model_segment_ids.append(instance_segment_ids)
        model_labels_ids.append(instance_labels_ids)

    return model_input_ids, model_segment_ids, model_input_mask, model_labels_ids


def get_dataloader(input_ids_lists, input_mask_lists, segment_ids_lists, labels_ids_lists, data_loader_shuffle=False):
    assert (len(input_ids_lists)
            == len(input_mask_lists)
            == len(segment_ids_lists)
            == len(labels_ids_lists))

    for i in range(len(input_ids_lists)):
        assert (len(input_ids_lists[i])
                == len(input_mask_lists[i])
                == len(segment_ids_lists[i])
                == len(labels_ids_lists[i]))

    dataset = JSONSpanDataset(
        input_ids_lists, input_mask_lists,
        segment_ids_lists, labels_ids_lists
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=data_loader_shuffle
    )

    return data_loader, len(input_ids_lists)


def get_relevance_dataloader(input_ids_lists, input_mask_lists, segment_ids_lists, labels_ids_lists, data_loader_shuffle=False):
    assert (len(input_ids_lists)
            == len(input_mask_lists)
            == len(segment_ids_lists)
            == len(labels_ids_lists))

    dataset = JSONRelevanceDataset(
        input_ids_lists, input_mask_lists,
        segment_ids_lists, labels_ids_lists
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=RELEVANCE_BATCH, shuffle=data_loader_shuffle
    )

    return data_loader, len(input_ids_lists)


# evaluation metrics
def find_tag_aware_spans(labels_sequence, blocks_ids_sequence=None):
    """
    Finds and returns spans with 'B'/'F' beginning within a sequence. A
    span starts with a "B"/"F" label, followed by contiguous "I" labels.
    Args:
        labels_sequence: List with labels.
        blocks_ids_sequence: List with block_id corresponding to each label.
    Returns:
        spans: Set of tuples. Each tuple indicates a span and stores contiguous
        indices and corresponding block_ids, and start label 'B'/'F' info.
    """
    spans = set()
    span_item = namedtuple('span_item', ['block_id', 'index', 'tag'])
    dummy_block_id = '_'

    i = 0
    while(i < len(labels_sequence)):
        if(labels_sequence[i] == 0
                or labels_sequence[i] == 3):
            if(labels_sequence[i] == 0):
                tag = 0
            else:
                tag = 3
            span = []

            if(blocks_ids_sequence):
                span.append(span_item(blocks_ids_sequence[i], i, tag))
            else:
                span.append(span_item(dummy_block_id, i, tag))

            i += 1

            while(i < len(labels_sequence) and labels_sequence[i] == 1):

                if(blocks_ids_sequence):
                    span.append(span_item(blocks_ids_sequence[i], i, tag))
                else:
                    span.append(span_item(dummy_block_id, i, tag))
                i += 1

            t = tuple(span)
            spans.add(t)

        else:
            i += 1

    return spans


def find_twostep_tag_aware_spans(labels_sequence):
    """
    Finds and returns spans with 'B'/'F' beginning within a sequence. A
    span starts with a "B"/"F" label, followed by contiguous "I" labels.
    Args:
        labels_sequence: List with labels (OR) List of tuples (label, block_index, block_offset).
    Returns:
        spans: Set of tuples. Each tuple indicates a span and stores contiguous
        indices and corresponding block_ids, and start label 'B'/'F' info.
    """
    spans = set()
    span_item = namedtuple('span_item', ['block_index', 'block_offset', 'tag'])

    i = 0
    while(i < len(labels_sequence)):
        if(labels_sequence[i][0] == 0
                or labels_sequence[i][0] == 3):
            if(labels_sequence[i][0] == 0):
                tag = 0
            else:
                tag = 3

            span = []
            span.append(span_item(labels_sequence[i][1],
                                  labels_sequence[i][2],
                                  tag))
            i += 1

            while(i < len(labels_sequence) and labels_sequence[i][0] == 1):
                span.append(span_item(labels_sequence[i][1],
                                      labels_sequence[i][2],
                                      tag))
                i += 1

            t = tuple(span)
            spans.add(t)
        else:
            i += 1

    return spans


class InstanceLevelMetrics:
    """
    This class returns exact match.
    """

    def __init__(self, for_single_model,
                 pruned_target_labels, target_labels, output_labels):
        """
        Args:
            for_single_model: True if evaluating single baseline model, else False.
            target_labels: List of target labels sequences. (List of lists).
            output_labels: List of model output labels sequences. (List of lists).
        """

        if pruned_target_labels is not None:
            assert len(pruned_target_labels) == len(output_labels)
        assert len(output_labels) == len(target_labels)

        self.for_single_model = for_single_model
        self.pruned_target_labels = pruned_target_labels
        self.target_labels = target_labels
        self.output_labels = output_labels

    def twostep_non_weighted_exact_match_complete_target(self):
        exact_match = 0
        examples = len(self.target_labels)

        for i in range(examples):
            target_spans = find_twostep_tag_aware_spans(self.target_labels[i])
            predicted_spans = find_twostep_tag_aware_spans(self.output_labels[i])

            if(target_spans == predicted_spans):
                exact_match += 1

        return exact_match / examples

    def non_weighted_exact_match_complete_target(self):
        exact_match = 0
        examples = len(self.target_labels)

        for i in range(examples):
            target_spans = find_tag_aware_spans(self.target_labels[i])
            predicted_spans = find_tag_aware_spans(self.output_labels[i])

            if(target_spans == predicted_spans):
                exact_match += 1

        exact_match = exact_match / examples

        return exact_match

    def return_all_metrics(self):
        """
        Returns:
            metrics: This is a dictionary with the required metrics.
        """
        if self.for_single_model:
            exact_match = self.non_weighted_exact_match_complete_target()
        else:
            exact_match = self.twostep_non_weighted_exact_match_complete_target()

        metrics = {"exact_match": exact_match}

        return metrics


def all_metrics_scores(for_single_model, target_labels, pruned_target_labels, output_labels):
    """
    This function returns all metric scores for span prediction.
    Args:
        for_single_model: True if evaluating single baseline model, else False.
        target_labels: List of complete target labels sequence. (List of lists)
        pruned_target_labels: List of pruned target labels sequence. (List of lists)
        output_labels: List of model output labels sequence. (List of lists)
    Returns:
        metrics: Dictionary with metrics.
    """
    instance_level_metrics_obj = InstanceLevelMetrics(for_single_model,
                                                      pruned_target_labels, target_labels,
                                                      output_labels)
    instance_level_metrics = instance_level_metrics_obj.return_all_metrics()

    return instance_level_metrics
