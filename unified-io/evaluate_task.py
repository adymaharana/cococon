import argparse
from os.path import exists
import os
import json
import sys
from PIL import Image
import spacy
import pprint
pp = pprint.PrettyPrinter(indent=4)

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
    return num_inconsistencies


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
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--image_dir", type=str)
    args = parser.parse_args()

    # open vqa file
    vqa = json.load(open(args.input_file))['annotations']
    print("Found %s samples" % len(vqa))

    image_dir = args.image_dir
    model = runner.ModelRunner(args.model_size, args.model_weights)

    all_predictions = []

    for i in tqdm(range(0, len(vqa), args.eval_batch_size)):

        predictions = [sample.copy() for sample in vqa[i:(i+args.eval_batch_size)]]

        # get images, queries
        images = []
        vqa_queries = []
        caption_queries = []
        loc_queries = []
        loc_idxs = []
        for j, sample in enumerate(vqa[i:(i+args.eval_batch_size)]):

            image_path = os.path.join(image_dir, 'COCO_' + sample["image"]["subset"] + '_' + str(sample["image"]["image_id"]).zfill(12) + '.jpg')

            with Image.open(image_path) as img:
                image = np.array(img)
                if len(image.shape) != 3 or image.shape[-1] != 3:
                    image = np.tile(np.expand_dims(image, axis=-1), (1, 1, 3))
                images.append(image)

            vqa_queries.append(sample["query"])
            caption_queries.append('What does the image describe ?')
            if 'detection' in sample:
                loc_queries.append(sample["detection"].strip().lower())
                loc_idxs.append(j)

        outputs = model.vqa(images, vqa_queries)
        print("**** VQA Output ****")
        print(outputs)

        for n, answer in enumerate(outputs['text']):
            predictions[n]['predict_answer'] = answer

        outputs = model.caption(images)
        print("**** Caption Output ****")
        print(outputs)

        for n, answer in enumerate(outputs['text']):
            predictions[n]['predict_caption'] = answer

        if len(loc_queries) > 0:
            for k, loc_query in zip(loc_idxs, loc_queries):
                try:
                    output = model.refexp(images[k], loc_query)
                    print("**** Referring Expression Output ****")
                    print(output)
                    predictions[k]['predict_bboxes'] = output['boxes'].tolist()
                except:
                    continue

        all_predictions.extend(predictions)
        if i % 5 == 0:
            with open(args.out_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)

    with open(args.out_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)

if __name__ == "__main__":
  main()
