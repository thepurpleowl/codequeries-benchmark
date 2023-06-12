from create_span_prediction_training_examples import PREPAREVOCAB
import dataset_with_context_pb2
from transformers import RobertaTokenizer
from tqdm import tqdm


def create_groupwise_prediction_dataset(block_query_subtokens_labels_dataset, vocab_file: str, model_type: str):
    """
    This function creates examples for groupwise relevance prediction dataset to be
    eventually used for span prediction.
    Args:
        block_query_subtokens_labels_dataset: BlockQuerySubtokensLabelsDataset protobuf
        vocab_file: model vocab file
        model_type: cubert/codebert
    Returns:
        ExampleForGroupwisePredictionDataset protobuf
    """
    prepare_vocab_object = PREPAREVOCAB(vocab_file)
    codebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    if(model_type == 'cubert'):
        cls_token = '[CLS]_'
        sep_token = '[SEP]_'
        cls_token_id = prepare_vocab_object.convert_by_vocab([cls_token])
        sep_token_id = prepare_vocab_object.convert_by_vocab([sep_token])
    elif(model_type == 'codebert'):
        cls_token = codebert_tokenizer.cls_token
        sep_token = codebert_tokenizer.sep_token
        cls_token_id = codebert_tokenizer.convert_tokens_to_ids([cls_token])
        sep_token_id = codebert_tokenizer.convert_tokens_to_ids([sep_token])

    example_for_group_pred = dataset_with_context_pb2.ExampleForGroupwisePredictionDataset()

    for i in tqdm(range(len(
            block_query_subtokens_labels_dataset.block_query_subtokens_labels_item)),
            desc="Groupwise_prediction"):

        single_group = dataset_with_context_pb2.SingleGroupExample()
        single_group.example_type = (block_query_subtokens_labels_dataset.
                                     example_types[i])
        single_group.distributable = (block_query_subtokens_labels_dataset.
                                      block_query_subtokens_labels_item[i].
                                      block_query_subtokens_labels_group_item[0].distributable)

        for k in range(len(
                block_query_subtokens_labels_dataset.block_query_subtokens_labels_item[i].
                block_query_subtokens_labels_group_item)):
            assert single_group.distributable == (block_query_subtokens_labels_dataset.
                                                  block_query_subtokens_labels_item[i].
                                                  block_query_subtokens_labels_group_item[k].distributable)

            group_item = dataset_with_context_pb2.SingleGroupItem()

            group_item.query_id = (block_query_subtokens_labels_dataset.
                                   block_query_subtokens_labels_item[i].
                                   block_query_subtokens_labels_group_item[k].query_id)

            group_item.block_id = (block_query_subtokens_labels_dataset.
                                   block_query_subtokens_labels_item[i].
                                   block_query_subtokens_labels_group_item[k].block.
                                   unique_block_id)

            group_item.relevance = (block_query_subtokens_labels_dataset.
                                    block_query_subtokens_labels_item[i].
                                    block_query_subtokens_labels_group_item[k].
                                    block.relevance_label)

            query_subtokens = []
            query_subtokens.extend(block_query_subtokens_labels_dataset.
                                   block_query_subtokens_labels_item[i].
                                   block_query_subtokens_labels_group_item[k].
                                   query_name_subtokens)

            program_subtokens = []
            labels = []

            for j in range(len(block_query_subtokens_labels_dataset.
                               block_query_subtokens_labels_item[i].
                               block_query_subtokens_labels_group_item[k].
                               block_subtokens_labels)):
                program_subtokens.append(block_query_subtokens_labels_dataset.
                                         block_query_subtokens_labels_item[i].
                                         block_query_subtokens_labels_group_item[k].
                                         block_subtokens_labels[j].program_subtoken)
                labels.append(block_query_subtokens_labels_dataset.
                              block_query_subtokens_labels_item[i].
                              block_query_subtokens_labels_group_item[k].
                              block_subtokens_labels[j].label)

            program_subtokens_ids = prepare_vocab_object.convert_by_vocab(
                program_subtokens)
            query_subtokens_ids = prepare_vocab_object.convert_by_vocab(
                query_subtokens)

            input_ids = []
            input_mask = []
            segment_ids = []
            labels_ids = []

            input_ids.extend(cls_token_id)
            segment_ids.append(0)
            input_mask.append(1)
            labels_ids.append(dataset_with_context_pb2.OutputLabels.Value("_"))

            for k in query_subtokens_ids:
                input_ids.append(k)
                segment_ids.append(0)
                input_mask.append(1)
                labels_ids.append(
                    dataset_with_context_pb2.OutputLabels.Value("_"))

            input_ids.extend(sep_token_id)
            segment_ids.append(0)
            input_mask.append(1)
            labels_ids.append(dataset_with_context_pb2.OutputLabels.Value("_"))

            for h, k in enumerate(program_subtokens_ids):
                input_ids.append(k)
                segment_ids.append(1)
                input_mask.append(1)
                labels_ids.append(labels[h])

            group_item.input_ids.extend(input_ids)
            group_item.input_mask.extend(input_mask)
            group_item.segment_ids.extend(segment_ids)
            group_item.label_ids.extend(labels_ids)
            group_item.program_ids.extend(program_subtokens_ids)
            group_item.program_label_ids.extend(labels)

            assert len(group_item.program_ids) == len(
                group_item.program_label_ids)

            single_group.group_items.append(group_item)

        single_group.query_name_token_ids.extend(query_subtokens_ids)

        example_for_group_pred.examples.append(single_group)

    return example_for_group_pred
