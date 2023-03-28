import argparse
from os.path import exists
import os
import json
import sys
from PIL import Image
import spacy
import pprint
pp = pprint.PrettyPrinter(indent=4)
from scipy.stats import spearmanr

from uio import runner
from uio.runner import IMAGE_GENERATION, REFEXP_PROMPT
from uio.configs import CONFIGS
import uio.utils as utils
import numpy as np

from absl import logging
import warnings
import random
import numpy as np
import pickle
from tqdm import tqdm

# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)
# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)
nlp = spacy.load('en_core_web_trf')
# IMAGE_GENERATION = 'What is the complete image? Text: " {} " .'

def rank(reference, reverse=False):
    sorted_reference = sorted(enumerate(reference), key=lambda t:t[1], reverse=reverse)
    return [idx for idx, val in sorted_reference]


def evaluate_contrast_sets(task_1_logprobs, task_2_logprobs):
    num_contrast_sets = len(task_2_logprobs)-1
    num_inconsistencies = [0] * num_contrast_sets
    baseline_task_1_logprob = task_1_logprobs[0]
    baseline_task_2_logprob = task_2_logprobs[0]
    for idx in range(1, num_contrast_sets+1):
        task_1_logprob = task_1_logprobs[idx]
        task_2_logprob = task_2_logprobs[idx]

        if (baseline_task_1_logprob < task_1_logprob and baseline_task_2_logprob > task_2_logprob) or (baseline_task_1_logprob > task_1_logprob and baseline_task_2_logprob < task_2_logprob):
            num_inconsistencies[idx-1] += 1
    rank_coeff = spearmanr(task_1_logprobs, task_2_logprobs).correlation
    return num_inconsistencies, rank_coeff


def get_open_img_annotations(data_file):
    with open(data_file) as f:
        data = json.load(f)
    id2info = {}
    id2captions = {}
    for s in data['images']:
        if s["id"] not in id2info:
            id2info[s["id"]] = s
    for s in data['annotations']:
        if s["image_id"] not in id2captions:
            id2captions[s["image_id"]] = []
            id2captions[s["image_id"]].append(s["caption"])
    return id2captions, id2info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_size", choices=list(CONFIGS))
    parser.add_argument("model_weights")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--image_dir", type=str)
    args = parser.parse_args()


    # open vqa file
    vqa = json.load(open(args.input_file))['annotations']
    print("Found %s samples" % len(vqa))
    image_dir = args.image_dir

    n_candidates = 5
    total_inconsistent_image = [0] * n_candidates
    total_inconsistent_vqa = [0] * n_candidates
    total_inconsistent_loc = [0] * n_candidates
    support_by_top_k = [0] * n_candidates
    support_by_top_k_loc = [0] * n_candidates
    image_correlations = []
    vqa_correlations = []
    loc_correlations = []

    model = runner.ModelRunner(args.model_size, args.model_weights)

    predictions = []
    total_vqa = 0

    for _, sample in tqdm(enumerate(vqa)):


        total_vqa += 1
        prediction = sample.copy()

        captions = []
        vqa_answer_candidates = []
        loc_descriptions = []
        vqa_answer_candidates.append(sample["answer"])
        captions.append(sample["caption"])

        if "detection" in sample:
            loc_descriptions.append(sample["detection"])


        contrast_set_idxs = []
        contrast_set_idxs_loc = []
        for k, c in enumerate(sample["contrast_sets"]):

            captions.append(c["mutex_caption"].lower())
            vqa_answer_candidates.append(c["mutex_answer"])
            contrast_set_idxs.append(k)

            if "detection" in sample and "detection" in c:
                contrast_set_idxs_loc.append(k)
                loc_descriptions.append(c["detection"])

        image_path = os.path.join(image_dir, 'COCO_val2014_' + str(sample["image_id"]).zfill(12) + '.jpg')

        with Image.open(image_path) as img:
            image = np.array(img)
            if len(image.shape) != 3 or image.shape[-1] != 3:
                image = np.tile(np.expand_dims(image, axis=-1), (1, 1, 3))

        # get captioning scores
        caption_scores = np.squeeze(model.run([image], ['What does the image describe ?'], answer_options=captions, average_loss=True)["all_scores"]).tolist()

        # get VQA scores
        vqa_scores = np.squeeze(model.run([image], [sample["query"]], answer_options=vqa_answer_candidates, average_loss=True)["all_scores"]).tolist()

        # text to image generation scores
        prompts = [IMAGE_GENERATION.replace("{}", description.strip().lower()) for description in captions]
        image_scores = np.squeeze(model.run([None]*len(captions), prompts, output_text_len=1, answer_options=[image], average_loss=True)["all_scores"]).tolist()

        # localization scores
        loc_scores = []
        if len(loc_descriptions) > 1:

            all_bbox_tokens = []
            for bbox in sample["boxes"]:
                correct_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bbox_tokens = utils.region_to_tokens(correct_bbox, image.shape[0], image.shape[1])
                all_bbox_tokens.append(' '.join(bbox_tokens))

            loc_prompts = [REFEXP_PROMPT.replace("{}", description.strip().lower()) for description in loc_descriptions]
            label_texts = [' %s ' % description.strip().lower() for description in loc_descriptions]
            loc_answer_options = [label_text.join(all_bbox_tokens) for label_text in label_texts]
            loc_scores = np.squeeze(model.run([image]*len(loc_prompts), loc_prompts, output_text_len=32, answer_options=loc_answer_options, average_loss=True)["all_scores"]).tolist()

        print("Caption Task Scores: ", caption_scores)
        print("VQA Task Scores: ", vqa_scores)
        print("Image Generation Task Scores", image_scores)

        prediction["target_image_loss"] = image_scores[0]
        prediction["target_caption_loss"] = caption_scores[0]
        prediction["target_answer_loss"] = vqa_scores[0]

        if len(loc_scores) > 1:
            caption_loc_scores = [caption_scores[0]] + [caption_scores[m+1] for m, idx in enumerate(contrast_set_idxs) if idx in contrast_set_idxs_loc]
            loc_caption_ranks = rank(caption_loc_scores[1:])
            caption_loc_scores = [caption_loc_scores[0]] + [caption_loc_scores[idx+1] for idx in loc_caption_ranks]
            loc_scores = [loc_scores[0]] + [loc_scores[idx+1] for idx in loc_caption_ranks]

        ranks = rank(caption_scores[1:])
        caption_scores = [caption_scores[0]] + [caption_scores[idx+1] for idx in ranks]
        image_scores = [image_scores[0]] + [image_scores[idx+1] for idx in ranks]
        vqa_scores = [vqa_scores[0]] + [vqa_scores[idx+1] for idx in ranks]

        num_inconsistencies_image, image_r = evaluate_contrast_sets(caption_scores, image_scores)
        image_correlations.append(image_r)
        for j, num in enumerate(num_inconsistencies_image):
            total_inconsistent_image[j] += num
            support_by_top_k[j] += 1
            prediction["contrast_sets"][contrast_set_idxs[ranks[j]]]["loss_image_contrast"] = image_scores[j + 1]
            prediction["contrast_sets"][contrast_set_idxs[ranks[j]]]["loss_caption_contrast"] = caption_scores[j + 1]

            if num == 1:
                prediction["contrast_sets"][contrast_set_idxs[ranks[j]]]["consistent_image"] = False
            else:
                prediction["contrast_sets"][contrast_set_idxs[ranks[j]]]["consistent_image"] = True

        num_inconsistencies_vqa, vqa_r = evaluate_contrast_sets(caption_scores, vqa_scores)
        vqa_correlations.append(vqa_r)
        for j, num in enumerate(num_inconsistencies_vqa):
            total_inconsistent_vqa[j] += num
            prediction["contrast_sets"][contrast_set_idxs[ranks[j]]]["loss_answer_contrast"] = vqa_scores[j + 1]

            if num == 1:
                prediction["contrast_sets"][contrast_set_idxs[ranks[j]]]["consistent_vqa"] = False
            else:
                prediction["contrast_sets"][contrast_set_idxs[ranks[j]]]["consistent_vqa"] = True

        # localization
        if len(loc_descriptions) > 1:
            print("Localization Scores: ", loc_scores)
            prediction["target_loc_loss"] = loc_scores[0]

            num_inconsistencies_loc, loc_r = evaluate_contrast_sets(caption_loc_scores, loc_scores)
            loc_correlations.append(loc_r)
            for j, num in enumerate(num_inconsistencies_loc):
                total_inconsistent_loc[j] += num
                prediction["contrast_sets"][contrast_set_idxs_loc[loc_caption_ranks[j]]]["loss_loc_contrast"] = loc_scores[j + 1]
                support_by_top_k_loc[j] += 1

                if num == 1:
                    prediction["contrast_sets"][contrast_set_idxs_loc[loc_caption_ranks[j]]]["consistent_loc"] = False

                else:
                    prediction["contrast_sets"][contrast_set_idxs_loc[loc_caption_ranks[j]]]["consistent_loc"] = True


        predictions.append(prediction)

        if total_vqa % 50 == 0:
            print("Total samples evaluated so far: ", total_vqa)
            with open(args.out_file, 'w') as f:
                json.dump(predictions, f, indent=2)


    print("Total samples evaluated so far: ", total_vqa)

    print("VQA Inconsistency Counts: ", total_inconsistent_vqa)
    for k, num in enumerate(total_inconsistent_vqa):
        print("Acc. top-%s: %s" % (k, round(float(num) / max(float(support_by_top_k[k]), 1), 2)))

    print("Image Generation Inconsistency Counts", total_inconsistent_image)
    for k, num in enumerate(total_inconsistent_image):
        print("Acc. top-%s: %s" % (k, round(float(num) / max(float(support_by_top_k[k]), 1), 2)))

    print("Localization Inconsistency Counts", total_inconsistent_loc)
    for k, num in enumerate(total_inconsistent_loc):
        print("Acc. top-%s: %s" % (k, round(float(num) / max(float(support_by_top_k_loc[k]), 1), 2)))

    print("Support", support_by_top_k)
    print("Support [Localization]", support_by_top_k_loc)

    print("VQA Spearmanr", np.mean(vqa_correlations), np.std(vqa_correlations))
    print("Image Spearmanr", np.mean(image_correlations), np.std(image_correlations))
    print("Loc Spearmanr", np.mean(loc_correlations), np.std(loc_correlations))

    with open(args.out_file, 'w') as f:
        json.dump(predictions, f, indent=2)


if __name__ == "__main__":
  main()
