import sys
import torch
import datasets
from absl import flags
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import DEVICE, Relevance_Classification_Model
from utils import get_relevance_dataloader_input, get_relevance_dataloader, eval_fn_relevance_prediction

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "vocab_file",
    "pretrained_model_configs/vocab.txt",
    "Path to Cubert vocabulary file."
)

flags.DEFINE_string(
    "model_checkpoint_path",
    "finetuned_ckpts/Twostep_Relevance-512",
    "Path to relevance model checkpoint."
)


def get_relevance_model_performance(vocab_file):
    examples_data = datasets.load_dataset("thepurpleowl/codequeries", "twostep",
                                          split=datasets.Split.TEST, use_auth_token=True)
    # evaluation
    model = Relevance_Classification_Model()
    model.to(DEVICE)
    model.load_state_dict(torch.load(FLAGS.model_checkpoint_path, map_location=DEVICE))

    model_input_ids, model_segment_ids, model_input_mask, model_labels_ids = get_relevance_dataloader_input(examples_data,
                                                                                                            vocab_file, False)
    eval_relevance_data_loader, _ = get_relevance_dataloader(
        model_input_ids,
        model_input_mask,
        model_segment_ids,
        model_labels_ids
    )

    eval_relevance_out, eval_relevance_targets = eval_fn_relevance_prediction(
        eval_relevance_data_loader, model, DEVICE, False)

    assert len(eval_relevance_targets) == len(eval_relevance_out)

    scores = precision_recall_fscore_support(eval_relevance_targets, eval_relevance_out)
    relevance_accuracy = accuracy_score(eval_relevance_targets, eval_relevance_out)
    # print(scores)
    print("Accuracy: ", relevance_accuracy, "Precision: ", scores[0][1], ", Recall: ", scores[1][1])


if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    get_relevance_model_performance(FLAGS.vocab_file)
