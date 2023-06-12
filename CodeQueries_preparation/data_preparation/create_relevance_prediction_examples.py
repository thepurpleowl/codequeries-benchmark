from create_span_prediction_training_examples import PREPAREVOCAB
import dataset_with_context_pb2
from transformers import RobertaTokenizer
from tqdm import tqdm


def create_relevance_prediction_examples(block_query_subtokens_labels_dataset, vocab_file: str,
                                         include_single_hop_examples, model_type):
    """
    This function creates examples for relevance prediction.
    Args:
        block_query_subtokens_labels_dataset: BlockQuerySubtokensLabelsDataset protobuf
        vocab_file: model vocab file
        include_single_hop_examples: True/False
        model_type: cubert/codebert
    Returns:
        ExampleforRelevancePredictionDataset protobuf
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

    example_for_model_dataset = dataset_with_context_pb2.ExampleforRelevancePredictionDataset()

    for i in tqdm(range(len(
            block_query_subtokens_labels_dataset.block_query_subtokens_labels_item)),
            desc="Relevance_dataset"):

        if include_single_hop_examples is False:
            if(block_query_subtokens_labels_dataset.block_query_subtokens_labels_item[i].
                    block_query_subtokens_labels_group_item[0].distributable == 1):
                continue

        for k in range(len(
                block_query_subtokens_labels_dataset.block_query_subtokens_labels_item[i].
                block_query_subtokens_labels_group_item)):

            example_for_model_dataset_item = dataset_with_context_pb2.ExampleforRelevancePrediction()
            example_for_model_dataset_item.example_type = (block_query_subtokens_labels_dataset.
                                                           example_types[i])

            query_subtokens = []
            query_subtokens.extend(block_query_subtokens_labels_dataset.
                                   block_query_subtokens_labels_item[i].
                                   block_query_subtokens_labels_group_item[k].
                                   query_name_subtokens)

            example_for_model_dataset_item.query_id = (block_query_subtokens_labels_dataset.
                                                       block_query_subtokens_labels_item[i].
                                                       block_query_subtokens_labels_group_item[k].
                                                       query_id)

            example_for_model_dataset_item.block_id = (block_query_subtokens_labels_dataset.
                                                       block_query_subtokens_labels_item[i].
                                                       block_query_subtokens_labels_group_item[k].
                                                       block.unique_block_id)

            example_for_model_dataset_item.relevance = (block_query_subtokens_labels_dataset.
                                                        block_query_subtokens_labels_item[i].
                                                        block_query_subtokens_labels_group_item[k].
                                                        block.relevance_label)

            example_for_model_dataset_item.program_path = (block_query_subtokens_labels_dataset.
                                                           block_query_subtokens_labels_item[i].
                                                           block_query_subtokens_labels_group_item[k].
                                                           raw_file.file_path.dataset_file_path.unique_file_path)

            program_subtokens = []

            for j in range(len(block_query_subtokens_labels_dataset.
                               block_query_subtokens_labels_item[i].
                               block_query_subtokens_labels_group_item[k].
                               block_subtokens_labels)):
                program_subtokens.append(block_query_subtokens_labels_dataset.
                                         block_query_subtokens_labels_item[i].
                                         block_query_subtokens_labels_group_item[k].
                                         block_subtokens_labels[j].program_subtoken)

            program_subtokens_ids = prepare_vocab_object.convert_by_vocab(
                program_subtokens)
            query_subtokens_ids = prepare_vocab_object.convert_by_vocab(
                query_subtokens)

            input_ids = []
            input_mask = []
            segment_ids = []

            input_ids.extend(cls_token_id)
            segment_ids.append(0)
            input_mask.append(1)

            for k in query_subtokens_ids:
                input_ids.append(k)
                segment_ids.append(0)
                input_mask.append(1)

            input_ids.extend(sep_token_id)
            segment_ids.append(0)
            input_mask.append(1)

            for k in program_subtokens_ids:
                input_ids.append(k)
                segment_ids.append(1)
                input_mask.append(1)

            example_for_model_dataset_item.input_ids.extend(input_ids)
            example_for_model_dataset_item.input_mask.extend(input_mask)
            example_for_model_dataset_item.segment_ids.extend(segment_ids)

            example_for_model_dataset.block_relevance_example.append(
                example_for_model_dataset_item)
    return example_for_model_dataset
