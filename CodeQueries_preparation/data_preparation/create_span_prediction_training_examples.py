import dataset_with_context_pb2
from transformers import RobertaTokenizer
from tqdm import tqdm
import hashlib


class PREPAREVOCAB:
    def __init__(self, vocab_file: str):
        self.vocab_file = vocab_file

    def load_vocab(self):
        """
        This function created a Cubert vocabulary to ids dictionary.
        """
        vocab = {}
        with open(self.vocab_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                vocab[line] = i
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


def create_span_prediction_training_examples(block_query_subtokens_labels_dataset, vocab_file: str,
                                             model_type: str):
    """
    This function creates examples for span prediction.
    Args:
        block_query_subtokens_labels_dataset: BlockQuerySubtokensLabelsDataset protobuf
        vocab_file: model vocab file
        model_type: cubert/codebert
    Returns:
        ExampleforSpanPredictionDataset protobuf
    """
    __DUMMY_BLOCK_ID__ = hashlib.md5('_'.encode("utf-8")).digest()

    prepare_vocab_object = PREPAREVOCAB(vocab_file)
    codebert_tokenizer = RobertaTokenizer.from_pretrained(
        "microsoft/codebert-base")

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

    example_for_model_dataset = dataset_with_context_pb2.ExampleforSpanPredictionDataset()

    for i in tqdm(range(len(
            block_query_subtokens_labels_dataset.block_query_subtokens_labels_item
    )), desc="Span_pred_dataset"):
        example_for_model_dataset_item = dataset_with_context_pb2.ExampleforSpanPrediction()
        example_for_model_dataset_item.example_type = (block_query_subtokens_labels_dataset.
                                                       example_types[i])

        query_subtokens = []
        query_subtokens.extend(block_query_subtokens_labels_dataset.
                               block_query_subtokens_labels_item[i].
                               block_query_subtokens_labels_group_item[0].
                               query_name_subtokens)

        example_for_model_dataset_item.query_id = (block_query_subtokens_labels_dataset.
                                                   block_query_subtokens_labels_item[i].
                                                   block_query_subtokens_labels_group_item[0].
                                                   query_id)

        example_for_model_dataset_item.distributable = (block_query_subtokens_labels_dataset.
                                                        block_query_subtokens_labels_item[i].
                                                        block_query_subtokens_labels_group_item[0].
                                                        distributable)

        program_subtokens = []
        labels = []
        blocks = []

        for k in range(len(
            block_query_subtokens_labels_dataset.block_query_subtokens_labels_item[i].
            block_query_subtokens_labels_group_item
        )):
            if(block_query_subtokens_labels_dataset.block_query_subtokens_labels_item[i].
               block_query_subtokens_labels_group_item[k].block.relevance_label == dataset_with_context_pb2.
               BlockRelevance.Value("yes")):

                block_id = (block_query_subtokens_labels_dataset.
                            block_query_subtokens_labels_item[i].
                            block_query_subtokens_labels_group_item[k].
                            block.unique_block_id)

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

                    blocks.append(block_id)

                program_subtokens.append(sep_token)
                labels.append(dataset_with_context_pb2.OutputLabels.Value("_"))
                blocks.append(__DUMMY_BLOCK_ID__)

        # Remove last sep token
        program_subtokens.pop(-1)
        labels.pop(-1)
        blocks.pop(-1)

        program_subtokens_ids = prepare_vocab_object.convert_by_vocab(
            program_subtokens)
        query_subtokens_ids = prepare_vocab_object.convert_by_vocab(
            query_subtokens)

        input_ids = []
        input_mask = []
        segment_ids = []
        labels_ids = []
        block_ids = []

        input_ids.extend(cls_token_id)
        segment_ids.append(0)
        input_mask.append(1)
        labels_ids.append(dataset_with_context_pb2.OutputLabels.Value("_"))
        block_ids.append(__DUMMY_BLOCK_ID__)

        for k in query_subtokens_ids:
            input_ids.append(k)
            segment_ids.append(0)
            input_mask.append(1)
            labels_ids.append(dataset_with_context_pb2.OutputLabels.Value("_"))
            block_ids.append(__DUMMY_BLOCK_ID__)

        input_ids.extend(sep_token_id)
        segment_ids.append(0)
        input_mask.append(1)
        labels_ids.append(dataset_with_context_pb2.OutputLabels.Value("_"))
        block_ids.append(__DUMMY_BLOCK_ID__)

        for h, k in enumerate(program_subtokens_ids):
            input_ids.append(k)
            segment_ids.append(1)
            input_mask.append(1)
            labels_ids.append(labels[h])
            block_ids.append(blocks[h])

        assert len(block_ids) == len(labels_ids)

        example_for_model_dataset_item.input_ids.extend(input_ids)
        example_for_model_dataset_item.input_mask.extend(input_mask)
        example_for_model_dataset_item.segment_ids.extend(segment_ids)
        example_for_model_dataset_item.labels_ids.extend(labels_ids)
        example_for_model_dataset_item.block_id.extend(block_ids)

        example_for_model_dataset.examples.append(
            example_for_model_dataset_item)
    return example_for_model_dataset
