"""
MS MARCO evaluation wrapper (BLEU / ROUGE / semantic similarity)

Usage:
  python evals/ms_marco_eval.py <path_to_reference_file> <path_to_candidate_file>

Note: this script depends on spacy and the ms_marco bleu/rouge modules. Install required
packages and the `bleu/` and `rouge/` modules in PYTHONPATH as described in the original
MS MARCO evaluation toolkit.
"""
from __future__ import print_function

import json
import sys
import spacy

from bleu.bleu import Bleu
from rouge.rouge import Rouge
from spacy.lang.en import English as NlpEnglish
nlp = spacy.load('en_core_web_lg')
QUERY_ID_JSON_ID = 'query_id'
ANSWERS_JSON_ID = 'answers'
NLP = None
MAX_BLEU_ORDER = 4

def normalize_batch(p_iter, p_batch_size=1000, p_thread_count=5):
    global NLP
    if not NLP:
        NLP = NlpEnglish(parser=False)

    output_iter = NLP.pipe(p_iter, \
                           batch_size=p_batch_size, \
                           n_threads=p_thread_count)

    for doc in output_iter:
        tokens = [str(w).strip().lower() for w in doc]
        yield ' '.join(tokens)

def load_file(p_path_to_data):
    all_answers = []
    query_ids = []
    no_answer_query_ids = set()
    with open(p_path_to_data, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            if not line.strip():
                continue
            try:
                json_object = json.loads(line)
            except json.JSONDecodeError:
                raise Exception('"%s" is not a valid json' % line)

            assert QUERY_ID_JSON_ID in json_object, 'json missing query_id'
            query_id = json_object[QUERY_ID_JSON_ID]

            assert ANSWERS_JSON_ID in json_object, 'json missing answers'
            answers = json_object[ANSWERS_JSON_ID]
            if 'No Answer Present.' in answers:
                no_answer_query_ids.add(query_id)
                answers = ['']
            all_answers.extend(answers)
            query_ids.extend([query_id]*len(answers))

    all_normalized_answers = normalize_batch(all_answers)

    query_id_to_answers_map = {}
    for i, normalized_answer in enumerate(all_normalized_answers):
        query_id = query_ids[i]
        if query_id not in query_id_to_answers_map:
            query_id_to_answers_map[query_id] = []
        query_id_to_answers_map[query_id].append(normalized_answer)
    return query_id_to_answers_map, no_answer_query_ids

def compute_metrics_from_files(p_path_to_reference_file,
                               p_path_to_candidate_file,
                               p_max_bleu_order):
    reference_dictionary, reference_no_answer_query_ids = load_file(p_path_to_reference_file)
    candidate_dictionary, candidate_no_answer_query_ids = load_file(p_path_to_candidate_file)
    query_id_answerable = set(reference_dictionary.keys())-reference_no_answer_query_ids
    query_id_answerable_candidate = set(candidate_dictionary.keys())-candidate_no_answer_query_ids
    
    true_positives = len(query_id_answerable_candidate.intersection(query_id_answerable))
    false_negatives = len(query_id_answerable)-true_positives
    true_negatives = len(candidate_no_answer_query_ids.intersection(reference_no_answer_query_ids))
    false_positives = len(reference_no_answer_query_ids)-true_negatives
    precision = float(true_positives)/(true_positives+false_positives) if (true_positives+false_positives)>0 else 1.
    recall = float(true_positives)/(true_positives+false_negatives) if (true_positives+false_negatives)>0 else 1.
    F1 = 2 *((precision*recall)/(precision+recall)) if (precision+recall)>0 else 0.0
    filtered_reference_dictionary = {key: value for key, value in reference_dictionary.items() if key not in reference_no_answer_query_ids}

    filtered_candidate_dictionary = {key: value for key, value in candidate_dictionary.items() if key not in reference_no_answer_query_ids}

    for query_id, answers in filtered_candidate_dictionary.items():
        assert len(answers) <= 1, 'query_id %d contains more than 1 answer' % query_id

    reference_query_ids = set(filtered_reference_dictionary.keys())
    candidate_query_ids = set(filtered_candidate_dictionary.keys())
    common_query_ids = reference_query_ids.intersection(candidate_query_ids)
    assert (len(common_query_ids) == len(reference_query_ids)) and (len(common_query_ids) == len(candidate_query_ids)), 'Reference and candidate files must share same query ids'

    all_scores = {}
    bleu_scores, _ = Bleu(p_max_bleu_order).compute_score(filtered_reference_dictionary, filtered_candidate_dictionary)
    for i, bleu_score in enumerate(bleu_scores):
        all_scores['bleu_%d' % (i+1)] = bleu_score

    rouge_score, _ = Rouge().compute_score(filtered_reference_dictionary, filtered_candidate_dictionary)
    all_scores['rouge_l'] = rouge_score
    all_scores['F1'] = F1
    similarity = 0
    for key in filtered_reference_dictionary:
        candidate_answer = nlp(filtered_candidate_dictionary[key][0])
        reference_answer = filtered_reference_dictionary[key]
        answersimilarity = 0
        for answer in reference_answer:
            answersimilarity += candidate_answer.similarity(nlp(answer))
        similarity += answersimilarity/len(reference_answer)
    semantic_similarity = similarity/len(filtered_reference_dictionary) if filtered_reference_dictionary else 0.0
    all_scores['Semantic_Similarity'] = semantic_similarity
    return all_scores

def main():
    path_to_reference_file = sys.argv[1]
    path_to_candidate_file = sys.argv[2]
    metrics = compute_metrics_from_files(path_to_reference_file, path_to_candidate_file, MAX_BLEU_ORDER)
    for metric in sorted(metrics):
        print('%s: %s' % (metric, metrics[metric]))

if __name__ == "__main__":
    main()
