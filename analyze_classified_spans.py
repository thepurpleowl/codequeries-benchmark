# %%
import pickle
from tqdm import tqdm
from collections import namedtuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from utils import get_examples_for_twostep, get_first_sep_label_index


with open('resources/twostep_TEST.pkl', 'rb') as f:
    examples_data = pickle.load(f)
twostep_dict, _ = get_examples_for_twostep(examples_data, example_types_to_evaluate='positive')
sep_token = '[SEP]_'
smoothing_fn = SmoothingFunction()
smoothing_variants = [smoothing_fn.method0, smoothing_fn.method1, smoothing_fn.method2, smoothing_fn.method3,
                      smoothing_fn.method4, smoothing_fn.method5, smoothing_fn.method6, smoothing_fn.method7]


NON_DISTRIBUTABLE_QUERIES = {
    'Unused import': 'UnusedImport',
    '`__eq__` not overridden when adding attributes': 'DefineEqualsWhenAddingAttributes',
    'Use of the return value of a procedure': 'UseImplicitNoneReturnValue',
    'Wrong number of arguments in a call': 'WrongNumberArgumentsInCall',
    'Comparison using is when operands support `__eq__`': 'IncorrectComparisonUsingIs',
    'Non-callable called': 'NonCallableCalled',
    '`__init__` method calls overridden method': 'InitCallsSubclassMethod',
    'Signature mismatch in overriding method': 'SignatureOverriddenMethod',
    'Conflicting attributes in base classes': 'ConflictingAttributesInBaseClasses',
    'Inconsistent equality and hashing': 'EqualsOrHash',
    'Flask app is run in debug mode': 'FlaskDebug',
    'Wrong number of arguments in a class instantiation': 'WrongNumberArgumentsInClassInstantiation',
    'Incomplete ordering': 'IncompleteOrdering',
    'Missing call to `__init__` during object initialization': 'MissingCallToInit',
    '`__iter__` method returns a non-iterator': 'IterReturnsNonIterator'
}


def analyze_relevance(prediction, target):
    with open(prediction, 'rb') as f:
        eval_out = pickle.load(f)
    with open(target, 'rb') as f:
        eval_target = pickle.load(f)
    scores = precision_recall_fscore_support(eval_target, eval_out)
    relevance_accuracy = accuracy_score(eval_target, eval_out)
    print({"Accuracy": relevance_accuracy, "Precision": scores[0][1], "Recall": scores[1][1]})

    with open('resources/twostep_TEST.pkl', 'rb') as f:
        all_blocks = pickle.load(f)

    query_wise_performance = {}
    for i, ex in enumerate(tqdm(all_blocks)):
        # if ex['query_name'] in query_wise_performance:
        #     query_wise_performance[ex['query_name']][0] += int(eval_out[i] == eval_target[i])
        #     query_wise_performance[ex['query_name']][1] += 1
        # else:
        #     query_wise_performance[ex['query_name']] = [int(eval_out[i] == eval_target[i]), 1]
        if ex['query_name'] in query_wise_performance:
            query_wise_performance[ex['query_name']][0].append(eval_out[i])
            query_wise_performance[ex['query_name']][1].append(eval_target[i])
        else:
            query_wise_performance[ex['query_name']] = [[eval_out[i]], [eval_target[i]]]

    querywise_scores = {}
    for query in query_wise_performance:
        # print(query, query_wise_performance[query][0] / query_wise_performance[query][1])
        scores = precision_recall_fscore_support(query_wise_performance[query][1], query_wise_performance[query][0])
        relevance_accuracy = accuracy_score(query_wise_performance[query][1], query_wise_performance[query][0])
        querywise_scores[query] = {"Accuracy": relevance_accuracy, "Precision": scores[0][1], "Recall": scores[1][1]}

    return querywise_scores


def find_twostep_tag_aware_spans(labels_sequence, span_type='both'):
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

    filtered_span = set()
    if span_type == 'answer':
        for span in spans:
            if span[0].tag == 0:
                filtered_span.add(span)
        return filtered_span
    elif span_type == 'sf':
        for span in spans:
            if span[0].tag == 3:
                filtered_span.add(span)
        return filtered_span
    elif span_type == 'both':
        return spans
    else:
        raise ValueError("Unknown span type")


# %%
def block_index_to_order_index(twostep_key, span_item):
    for example_index in twostep_dict[twostep_key]:
        block = examples_data[example_index]
        if block['context_block']['index'] == span_item.block_index:
            first_sep_index = get_first_sep_label_index(block['label_sequence'])
            assert block['subtokenized_input_sequence'][first_sep_index] == sep_token

            return block['subtokenized_input_sequence'][first_sep_index + 1:][span_item.block_offset]
    return None


def get_span_subtokens(twostep_key, spans):
    # span_item = namedtuple('span_item', ['block_index', 'block_offset', 'tag'])
    ans_span_i = 0
    sf_span_i = 0
    subtokens_of_spans = []
    for span in spans:
        span_subtokens = []

        for span_item in span:
            span_subtoken = block_index_to_order_index(twostep_key, span_item)
            if span_subtoken:
                # if span index not marked
                if not span_subtokens:
                    if span_item.tag == 0:
                        span_subtokens.append('ans' + str(ans_span_i))
                        ans_span_i += 1
                    elif span_item.tag == 3:
                        span_subtokens.append('sf' + str(sf_span_i))
                        sf_span_i += 1
                    else:
                        raise
                span_subtokens.append(span_subtoken)
        # single sentence from all the spans
        subtokens_of_spans.extend(span_subtokens)
    return subtokens_of_spans


def get_formatted_bleu(twostep_key, target_labels, output_labels, span_type='both', smoothing=0):
    target_spans = find_twostep_tag_aware_spans(target_labels, span_type=span_type)
    predicted_spans = find_twostep_tag_aware_spans(output_labels, span_type=span_type)

    # sort spans a/q block index, sorting by block offset implicitly attained
    target_spans = sorted(target_spans, key=lambda x: (x[0].block_index))
    predicted_spans = sorted(predicted_spans, key=lambda x: (x[0].block_index))

    reference = get_span_subtokens(twostep_key, target_spans)
    candidate = get_span_subtokens(twostep_key, predicted_spans)
    # print(twostep_key, reference, candidate)

    # references = [[['my', 'first', 'incorrect', 'sentence']]]
    # candidates = [['my', 'incorrect', 'sentence']]
    # for sv in smoothing_variants:
    #     score = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sv)
    #     print(score)
    bleu_score = corpus_bleu([[reference]], [candidate],
                             weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smoothing_variants[smoothing])

    return reference, candidate, bleu_score


def twostep_non_weighted_exact_match_complete_target(target_labels, output_labels, span_type='both'):
    target_spans = find_twostep_tag_aware_spans(target_labels, span_type=span_type)
    predicted_spans = find_twostep_tag_aware_spans(output_labels, span_type=span_type)

    if(target_spans == predicted_spans):
        return 1

    return 0


def analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction,
                           smoothing=0, example_type='positive', span_type='both'):
    with open(twostep_meta, 'rb') as f:
        twostep_meta = pickle.load(f)
    with open(twostep_order, 'rb') as f:
        twostep_order = pickle.load(f)
    with open(twostep_target, 'rb') as f:
        twostep_target = pickle.load(f)
    with open(twostep_prediction, 'rb') as f:
        twostep_prediction = pickle.load(f)

    query_wise_performance = {}
    references, candidates = [], []
    for i, ex in enumerate(tqdm(twostep_order)):
        # assert twostep_meta[ex] == (example_type == 'negative')
        em = twostep_non_weighted_exact_match_complete_target(twostep_target[i], twostep_prediction[i],
                                                              span_type=span_type)
        # if em == 1 and 'Inconsistent' in ex[0]:
        #     print(ex)

        # bleu score
        reference, candidate, sent_bleu_score = get_formatted_bleu(ex, twostep_target[i], twostep_prediction[i],
                                                                   span_type=span_type,
                                                                   smoothing=smoothing)
        references.append([reference])  # single reference
        candidates.append(candidate)
        if ex[0] in query_wise_performance:
            query_wise_performance[ex[0]][0] += em
            query_wise_performance[ex[0]][1] += 1
            query_wise_performance[ex[0]][2] += sent_bleu_score
        else:
            query_wise_performance[ex[0]] = [em, 1, sent_bleu_score]

    # get micro averaged bleu
    bleu_score = corpus_bleu(references, candidates,
                             weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smoothing_variants[smoothing])

    all_em = 0
    all_sent_bleu = 0
    total = 0
    querywise_scores = {}
    for query in query_wise_performance:
        # print(query, query_wise_performance[query][0] / query_wise_performance[query][1])
        all_em += query_wise_performance[query][0]
        all_sent_bleu += query_wise_performance[query][2]
        total += query_wise_performance[query][1]
        querywise_scores[query] = (query_wise_performance[query][0] / query_wise_performance[query][1])
    print('EM: ', all_em / total, ' Corpus BLEU: ', bleu_score, ' Sent BLEU: ', all_sent_bleu / total)

    return querywise_scores


# %%
smoothing_i = 2
span_type = 'answer'
twostep_meta = 'analyses/twostep_sampled_positive_model_model_meta_dict.pkl'
twostep_order = 'analyses/twostep_sampled_positive_model_model_example_order.pkl'
twostep_target = 'analyses/model_sampled_positive_model_model_target_metadata_ids.pkl'
twostep_prediction = 'analyses/model_sampled_positive_model_model_predicted_labels_ids.pkl'
all_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction,
                                             smoothing=smoothing_i, span_type=span_type)

twostep_meta = 'analyses/twostep_sampled_positive_model_finet_meta_dict.pkl'
twostep_order = 'analyses/twostep_sampled_positive_model_finet_example_order.pkl'
twostep_target = 'analyses/model_sampled_positive_model_finet_target_metadata_ids.pkl'
twostep_prediction = 'analyses/model_sampled_positive_model_finet_predicted_labels_ids.pkl'
all_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction,
                                             smoothing=smoothing_i, span_type=span_type)

twostep_meta = 'analyses/twostep_sampled_positive_finet_model_meta_dict.pkl'
twostep_order = 'analyses/twostep_sampled_positive_finet_model_example_order.pkl'
twostep_target = 'analyses/model_sampled_positive_finet_model_target_metadata_ids.pkl'
twostep_prediction = 'analyses/model_sampled_positive_finet_model_predicted_labels_ids.pkl'
all_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction,
                                             smoothing=smoothing_i, span_type=span_type)

twostep_meta = 'analyses/twostep_sampled_positive_finet_finet_meta_dict.pkl'
twostep_order = 'analyses/twostep_sampled_positive_finet_finet_example_order.pkl'
twostep_target = 'analyses/model_sampled_positive_finet_finet_target_metadata_ids.pkl'
twostep_prediction = 'analyses/model_sampled_positive_finet_finet_predicted_labels_ids.pkl'
all_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction,
                                             smoothing=smoothing_i, span_type=span_type)


# %%

twostep_meta = 'analyses/twostep_sampled_negative_model_model_meta_dict.pkl'
twostep_order = 'analyses/twostep_sampled_negative_model_model_example_order.pkl'
twostep_target = 'analyses/model_sampled_negative_model_model_target_metadata_ids.pkl'
twostep_prediction = 'analyses/model_sampled_negative_model_model_predicted_labels_ids.pkl'
all_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction)

twostep_meta = 'analyses/twostep_sampled_negative_model_finet_meta_dict.pkl'
twostep_order = 'analyses/twostep_sampled_negative_model_finet_example_order.pkl'
twostep_target = 'analyses/model_sampled_negative_model_finet_target_metadata_ids.pkl'
twostep_prediction = 'analyses/model_sampled_negative_model_finet_predicted_labels_ids.pkl'
all_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction)

twostep_meta = 'analyses/twostep_sampled_negative_finet_model_meta_dict.pkl'
twostep_order = 'analyses/twostep_sampled_negative_finet_model_example_order.pkl'
twostep_target = 'analyses/model_sampled_negative_finet_model_target_metadata_ids.pkl'
twostep_prediction = 'analyses/model_sampled_negative_finet_model_predicted_labels_ids.pkl'
all_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction)

twostep_meta = 'analyses/twostep_sampled_negative_finet_finet_meta_dict.pkl'
twostep_order = 'analyses/twostep_sampled_negative_finet_finet_example_order.pkl'
twostep_target = 'analyses/model_sampled_negative_finet_finet_target_metadata_ids.pkl'
twostep_prediction = 'analyses/model_sampled_negative_finet_finet_predicted_labels_ids.pkl'
all_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction)


# %%
import sys
if __name__ == '__main__':
    argv = sys.argv
    if argv[1] == 'rel':
        prediction = 'analyses/eval_relevance_out_10.pkl'
        target = 'analyses/eval_relevance_targets_10.pkl'

        mini_scores = analyze_relevance(prediction, target)

        prediction = 'analyses/eval_relevance_out_all.pkl'
        target = 'analyses/eval_relevance_targets_all.pkl'

        all_scores = analyze_relevance(prediction, target)

        i_sh = 0
        i_mh = 0
        for query in all_scores:
            if all_scores[query]['Recall'] - mini_scores[query]['Recall'] > 0.2:
                print(query, all_scores[query], all_scores[query]['Recall'] - mini_scores[query]['Recall'])
            val = all_scores[query]['Accuracy'] - mini_scores[query]['Accuracy']
            mh = 'mh' if query in NON_DISTRIBUTABLE_QUERIES else 'sh'
            if val > 0.10:
                # print('max ', mh, query, mini_scores[query]['Accuracy'], val)
                if mh == 'mh':
                    i_mh += 1
                else:
                    i_sh += 1
            if val < 0.015:
                # print('min ', query, mini_scores[query]['Accuracy'], val)
                pass

        sh = []
        mh = []
        for query in all_scores:
            val = all_scores[query]['Accuracy'] - mini_scores[query]['Accuracy']
            if query in NON_DISTRIBUTABLE_QUERIES:
                mh.append(val)
            else:
                sh.append(val)
        assert len(sh) == 37
        assert len(mh) == 15
        print(sum(sh) / len(sh), sum(mh) / len(mh))
        print(i_sh, i_mh)

    else:
        if argv[2] == '10_10':
            # twostep_meta = 'analyze_sampled/twostep_positive_model_model_meta_dict.pkl'
            # twostep_order = 'analyze_sampled/twostep_positive_model_model_example_order.pkl'
            # twostep_target = 'analyze_sampled/model_positive_model_model_target_metadata_ids.pkl'
            # twostep_prediction = 'analyze_sampled/model_positive_model_model_predicted_labels_ids.pkl'
            twostep_meta = 'analyses/twostep_positive_model_model_meta_dict.pkl'
            twostep_order = 'analyses/twostep_positive_model_model_example_order.pkl'
            twostep_target = 'analyses/model_positive_model_model_target_metadata_ids.pkl'
            twostep_prediction = 'analyses/model_positive_model_model_predicted_labels_ids.pkl'

            zero = 0
            querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction)
            for query in querywise_score:
                if querywise_score[query] > 0.15:
                    print(query, querywise_score[query])
                else:
                    zero += 1
            print(zero)
        else:

            twostep_meta = 'analyses/twostep_positive_finet_finet_meta_dict.pkl'
            twostep_order = 'analyses/twostep_positive_finet_finet_example_order.pkl'
            twostep_target = 'analyses/model_positive_finet_finet_target_metadata_ids.pkl'
            twostep_prediction = 'analyses/model_positive_finet_finet_predicted_labels_ids.pkl'
            all_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction)

            twostep_meta = 'analyses/twostep_positive_finet_model_meta_dict.pkl'
            twostep_order = 'analyses/twostep_positive_finet_model_example_order.pkl'
            twostep_target = 'analyses/model_positive_finet_model_target_metadata_ids.pkl'
            twostep_prediction = 'analyses/model_positive_finet_model_predicted_labels_ids.pkl'
            mini_querywise_score = analyze_spanprediction(twostep_meta, twostep_order, twostep_target, twostep_prediction)

            i_sh, i_mh = 0, 0
            for query in mini_querywise_score:
                if mini_querywise_score[query] > 0.5:
                    print(query, mini_querywise_score[query])
                val = all_querywise_score[query] - mini_querywise_score[query]
                mh = 'mh' if query in NON_DISTRIBUTABLE_QUERIES else 'sh'
                if val > 0.50:
                    print('max ', mh, query, mini_querywise_score[query], val)
                    if mh == 'mh':
                        i_mh += 1
                    else:
                        i_sh + 1
                if val < 0.015:
                    # print('min ', query, mini_scores[query], val)
                    pass

            sh = []
            mh = []
            for query in mini_querywise_score:
                val = all_querywise_score[query] - mini_querywise_score[query]
                if query in NON_DISTRIBUTABLE_QUERIES:
                    mh.append(val)
                else:
                    sh.append(val)
            assert len(sh) == 37
            assert len(mh) == 15
            print(sum(sh) / len(sh), sum(mh) / len(mh))
