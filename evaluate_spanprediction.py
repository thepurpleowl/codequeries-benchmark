import sys
from absl import flags
import torch
import datasets
import csv
from utils import Cubert_Model, MAX_LEN
from utils import get_dataloader_input, get_twostep_dataloader_input, get_dataloader
from utils import prepare_sliding_window_input, prepare_sliding_window_output
from utils import eval_fn, all_metrics_scores

FLAGS = flags.FLAGS
DEVICE = "cuda:6"

flags.DEFINE_string(
    "example_types_to_evaluate",
    "positive",
    "'positive'/'negative'/'all'"
)

flags.DEFINE_string(
    "span_model_checkpoint_path",
    "finetuned_ckpts/Cubert-1K",
    "Path to span prediction model checkpoint."
)

flags.DEFINE_string(
    "relevance_model_checkpoint_path",
    "finetuned_ckpts/Twostep_Relevance-512",
    "Path to relevance prediction model checkpoint."
)

flags.DEFINE_string(
    "results_store_path",
    "codequeries_results.csv",
    "Path to csv where to store results."
)

flags.DEFINE_string(
    "vocab_file",
    "pretrained_model_configs/vocab.txt",
    "Path to Cubert vocabulary file."
)

flags.DEFINE_string(
    "setting",
    "ideal",
    "ideal/file_ideal/prefix/sliding_window/twostep"
)

flags.DEFINE_integer(
    "sliding_window_width",
    1024,
    "Window size for `sliding_window` setting."
)

flags.DEFINE_string(
    "span_type",
    "both",
    "both/answer/sf"
)


if __name__ == "__main__":
    argv = FLAGS(sys.argv)

    # write headers to results file
    HEADER = ["Model Name", "Setting", "Example type", "Exact_match"]
    if FLAGS.setting == "sliding_window":
        examples_data = datasets.load_dataset("thepurpleowl/codequeries", "prefix",
                                              split=datasets.Split.TEST, use_auth_token=True)
    else:
        examples_data = datasets.load_dataset("thepurpleowl/codequeries", FLAGS.setting,
                                              split=datasets.Split.TEST, use_auth_token=True)

    if(FLAGS.setting != "twostep"):
        if(FLAGS.setting == "sliding_window"):
            (model_input_ids, model_segment_ids,
             model_input_mask, model_labels_ids) = get_dataloader_input(examples_data,
                                                                        FLAGS.example_types_to_evaluate,
                                                                        "prefix",
                                                                        FLAGS.vocab_file)
            (model_input_ids, model_segment_ids,
             model_input_mask, model_labels_ids) = prepare_sliding_window_input(model_input_ids, model_segment_ids,
                                                                                model_input_mask, model_labels_ids,
                                                                                FLAGS.sliding_window_width)
            target_sequences = model_labels_ids
            sw_labels_ids = [x[2] for x in model_labels_ids]
            eval_data_loader, eval_file_length = get_dataloader(
                model_input_ids,
                model_input_mask,
                model_segment_ids,
                sw_labels_ids
            )
        else:
            (model_input_ids, model_segment_ids,
             model_input_mask, model_labels_ids) = get_dataloader_input(examples_data,
                                                                        FLAGS.example_types_to_evaluate,
                                                                        FLAGS.setting,
                                                                        FLAGS.vocab_file)
            target_sequences = model_labels_ids
            eval_data_loader, eval_file_length = get_dataloader(
                model_input_ids,
                model_input_mask,
                model_segment_ids,
                model_labels_ids
            )

        # evaluation
        model = Cubert_Model()
        model.to(DEVICE)
        model.load_state_dict(torch.load(FLAGS.span_model_checkpoint_path, map_location=DEVICE))

        pruned_target_sequences, output_sequences, _ = eval_fn(
            eval_data_loader, model, DEVICE)

        pruned_target_sequences = pruned_target_sequences.tolist()
        output_sequences = output_sequences.tolist()

        if(FLAGS.setting == "sliding_window"):
            (target_sequences, pruned_target_sequences,
             output_sequences) = prepare_sliding_window_output(target_sequences, pruned_target_sequences,
                                                               output_sequences, FLAGS.sliding_window_width)

        metrics = all_metrics_scores(True, target_sequences,
                                     pruned_target_sequences, output_sequences,
                                     FLAGS.span_type)
        RESULTS = ['CuBERT-1K', FLAGS.setting, FLAGS.example_types_to_evaluate, metrics["exact_match"]]
    else:
        from utils import Relevance_Classification_Model
        # evaluation
        relevance_model = Relevance_Classification_Model()
        span_model = Cubert_Model()
        relevance_model.to(DEVICE)
        span_model.to(DEVICE)
        relevance_model.load_state_dict(torch.load(FLAGS.relevance_model_checkpoint_path, map_location=DEVICE))
        span_model.load_state_dict(torch.load(FLAGS.span_model_checkpoint_path, map_location=DEVICE))

        (model_input_ids, model_segment_ids, model_input_mask, model_labels_ids,
         model_label_metadata_ids, model_target_metadata_ids) = get_twostep_dataloader_input(examples_data,
                                                                                             FLAGS.example_types_to_evaluate,
                                                                                             FLAGS.vocab_file,
                                                                                             relevance_model,
                                                                                             DEVICE)

        eval_data_loader, eval_file_length = get_dataloader(
            model_input_ids,
            model_input_mask,
            model_segment_ids,
            model_labels_ids
        )

        _, output_sequences, _ = eval_fn(
            eval_data_loader, span_model, DEVICE)

        output_sequences = output_sequences.tolist()
        # no of examples
        assert len(output_sequences) == len(model_label_metadata_ids)

        model_predicted_labels_ids = []
        for i, instance_output_sequence in enumerate(output_sequences):
            instance_predicted_labels_ids = []
            predicted_relevant_subtokens = min(len(model_label_metadata_ids[i]), MAX_LEN)
            for j in range(predicted_relevant_subtokens):
                instance_predicted_labels_ids.append((instance_output_sequence[j],
                                                      model_label_metadata_ids[i][j][1],
                                                      model_label_metadata_ids[i][j][2]))
            model_predicted_labels_ids.append(instance_predicted_labels_ids)
        # no of examples
        assert len(model_predicted_labels_ids) == len(model_target_metadata_ids)

        metrics = all_metrics_scores(False, model_target_metadata_ids,
                                     None, model_predicted_labels_ids,
                                     FLAGS.span_type)

        RESULTS = ['CuBERT-1K', FLAGS.setting, FLAGS.example_types_to_evaluate, metrics["exact_match"]]

    assert len(HEADER) == len(RESULTS)

    with open(FLAGS.results_store_path, "a") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(HEADER)

        csvwriter = csv.writer(f)
        csvwriter.writerow(RESULTS)
