import os
import numpy as np
from collections import Counter
import re
import json
import object_detection.lib.Evaluator as det_evaluator
from tqdm import tqdm
from PIL import Image

task_to_id = {
    'CocoVqa': 'question_id',
    'CocoClassification': 'id',
    'CocoCaptioning': 'cap_id',
    'CocoDetection': 'id',
    'RefCocop': 'sent_id',
}

img_dir = '/nas-ssd/adyasha/datasets/mscoco2014/val2014/'

def read_image(filename):
    img = Image.open(filename)
    try:
        data = np.asarray(img, dtype='uint8')
    except SystemError:
        data = np.asarray(img.getdata(), dtype='uint8')
    return data

class CocoEval():
    def __init__(self, samples, predictions, boxes, task):
        self.task = task
        self.task_id_name = task_to_id[self.task]
        self.samples = {str(s[self.task_id_name]): s for s in samples}
        self.predictions = predictions
        self.boxes = boxes

    def sample_novelty(self, sample):
        if len(sample['coco_categories']['unseen']) > 0:
            return 'held_out_concepts'

        return 'seen_concepts'


class CocoVqa(CocoEval):
    def __init__(self, samples, predictions, boxes, task='CocoVqa'):
        super().__init__(samples, predictions, boxes, task)

        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                             "couldnt": "couldn't", \
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't",
                             "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                             "hed": "he'd", "hed've": "he'd've", \
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                             "Id've": "I'd've", "I'dve": "I'd've", \
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                             "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
                             "mightn'tve": "mightn't've", "mightve": "might've", \
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                             "oclock": "o'clock", "oughtnt": "oughtn't", \
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                             "shed've": "she'd've", "she'dve": "she'd've", \
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                             "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                             "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                             "someone'dve": "someone'd've", \
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                             "somethingd've": "something'd've", \
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                             "thered": "there'd", "thered've": "there'd've", \
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                             "theyd've": "they'd've", \
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                             "twas": "'twas", "wasnt": "wasn't", \
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
                             "whatll": "what'll", "whatre": "what're", \
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                             "wheres": "where's", "whereve": "where've", \
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
                             "whos": "who's", "whove": "who've", "whyll": "why'll", \
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                             "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                             "yall'd've": "y'all'd've", \
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
                             "youd've": "you'd've", "you'dve": "you'd've", \
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap = {'none': '0',
                          'zero': '0',
                          'one': '1',
                          'two': '2',
                          'three': '3',
                          'four': '4',
                          'five': '5',
                          'six': '6',
                          'seven': '7',
                          'eight': '8',
                          'nine': '9',
                          'ten': '10'
                          }
        self.articles = ['a',
                         'an',
                         'the'
                         ]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                       outText,
                                       re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def preprocess_vqa_answer(self, answer):
        answer = answer.replace('\n', ' ')
        answer = answer.replace('\t', ' ')
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        return answer

    def evaluate(self, novelty='everything'):
        absent = 0
        correct = {'all': 0}
        total = {'all': 0}
        for anno_type in ['answer_type', 'question_type']:
            correct[anno_type] = Counter()
            total[anno_type] = Counter()

        for k, sample in self.samples.items():
            if novelty is not 'everything' and \
                    self.sample_novelty(sample) != novelty:
                continue

            if k not in self.predictions:
                absent += 1
                continue

            pred_answer = self.preprocess_vqa_answer(self.predictions[k]['answer'].lower())
            gt_answers = {self.preprocess_vqa_answer(k.lower()): v for k, v in sample['all_answers'].items()}
            answer_type = sample['anno']['answer_type']
            question_type = sample['anno']['question_type']
            if pred_answer in gt_answers:
                correctness = min(gt_answers[pred_answer] / 3, 1)
                correct['all'] += correctness
                correct['answer_type'][answer_type] += correctness
                correct['question_type'][question_type] += correctness

            total['all'] += 1
            total['answer_type'][answer_type] += 1
            total['question_type'][question_type] += 1

        eps = 1e-6
        accuracy = {'all': round(100 * correct['all'] / (eps + total['all']), 2)}
        accuracy['answer_type'] = {
            a: round(100 * correct['answer_type'][a] / (eps + total['answer_type'][a]), 2)
            for a in total['answer_type']}
        accuracy['question_type'] = {
            a: round(100 * correct['question_type'][a] / (eps + total['question_type'][a]), 2)
            for a in total['question_type']}
        metrics = {
            'correct': correct,
            'total': total,
            'absent': absent,
            'accuracy': accuracy,
        }

        return metrics


class CocoDetection(CocoEval):
    def __init__(self, samples, predictions, boxes, task='CocoDetection'):
        super().__init__(samples, predictions, boxes, task)

    def evaluate(self, novelty='everything', iou_thresh=0.5):
        absent = 0
        correct = Counter()
        total = Counter()
        APs = []
        eval_engine = det_evaluator.Evaluator()
        print(len(self.predictions.keys()), len(self.boxes.keys()))
        for k, sample in tqdm(self.samples.items()):
            if novelty is not 'everything' and \
                    self.sample_novelty(sample) != novelty:
                continue

            if k not in self.predictions:
                absent += 1
                continue

            scores = self.boxes[k]['relevance'][()]  # probs
            pred_boxes = self.boxes[k]['boxes'][()]  # scaled 0 to 1 (cx,cy,w,h)
            if pred_boxes.shape[0] == 0:
                absent += 1
                continue
            # convert to (x,y,w,h)
            # pred_boxes[:, 0] = pred_boxes[:, 0] - 0.5 * pred_boxes[:, 2]
            # pred_boxes[:, 1] = pred_boxes[:, 1] - 0.5 * pred_boxes[:, 3]
            # conert from x0, y0, x1, y1 to x, y, w, h

            pred_boxes[:, 2] = pred_boxes[:, 2] - pred_boxes[:, 0]
            pred_boxes[:, 3] = pred_boxes[:, 3] - pred_boxes[:, 1]

            try:
                gt_boxes = sample['original_boxes']  # absolute coord (x,y,w,h)
                if type(gt_boxes[0]) in [float, int]:
                    gt_boxes = np.array([gt_boxes])
                else:
                    gt_boxes = np.array(gt_boxes)
            except KeyError:
                continue
            # convert to relative coordinates
            # W = sample['image']['W']
            # H = sample['image']['H']
            image = read_image(os.path.join(img_dir, 'COCO_val2014_%s.jpg' % str(sample['image']['image_id']).zfill(12)))
            # print(image.shape)
            W = image.shape[0]
            H = image.shape[1]
            pred_boxes[:, 0] = pred_boxes[:, 0] / W
            pred_boxes[:, 1] = pred_boxes[:, 1] / H
            pred_boxes[:, 2] = pred_boxes[:, 2] / W
            pred_boxes[:, 3] = pred_boxes[:, 3] / H
            # print(gt_boxes.shape, sample['original_boxes'])
            gt_boxes[:, 0] = gt_boxes[:, 0] / W
            gt_boxes[:, 1] = gt_boxes[:, 1] / H
            gt_boxes[:, 2] = gt_boxes[:, 2] / W
            gt_boxes[:, 3] = gt_boxes[:, 3] / H
            B = pred_boxes.shape[0]
            all_boxes = det_evaluator.BoundingBoxes()
            for b in range(B):
                x, y, w, h = pred_boxes[b]
                all_boxes.addBoundingBox(det_evaluator.BoundingBox(
                    imageName=sample['image']['image_id'],
                    classId=sample['original_detection'],
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=det_evaluator.CoordinatesType.Relative,
                    imgSize=(W, H),
                    bbType=det_evaluator.BBType.Detected,
                    classConfidence=scores[b],
                    format=det_evaluator.BBFormat.XYWH))

            B = gt_boxes.shape[0]
            for b in range(B):
                x, y, w, h = gt_boxes[b]
                all_boxes.addBoundingBox(det_evaluator.BoundingBox(
                    imageName=sample['image']['image_id'],
                    classId=sample['original_detection'],
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=det_evaluator.CoordinatesType.Relative,
                    imgSize=(W, H),
                    bbType=det_evaluator.BBType.GroundTruth,
                    format=det_evaluator.BBFormat.XYWH))

            det_metrics = eval_engine.GetPascalVOCMetrics(all_boxes, iou_thresh)
            APs.append(det_metrics[0]['AP'])
            total['all'] += 1
            total[sample['original_detection']] += 1

        # mAP = np.mean(APs)
        mAP = np.sum(APs)/len(self.samples)

        metrics = {
            'absent': absent,
            'total': total,
            'mAP': mAP
        }

        return metrics


if __name__ == "__main__":

    gts = json.load(open('../temp/cococon_new_loc_v1.json'))
    for i in range(len(gts)):
        gts[i]['id'] = i+1

    # vqa UnifiedIO
    # for size in ['small', 'base', 'large', 'xl']:
    #     print(size)
    #     refs = json.load(open('./uio/cococon_task_outputs_%s.json' % size))
    #     refs = {str(s['question_id']): s for s in refs}
    #     coco = CocoVqa(gts, refs, None)
    #     results = coco.evaluate()
    #     for k, v in results.items():
    #         if k == 'accuracy':
    #             print(k)
    #             print(v)

    # VQA OFA
    for size in ['medium', 'base', 'large', 'huge', 'ofacon']:
        print(size)
        refs = json.load(open('../results/vqa/_predict_%s.json' % size))
        refs = {str(s['question_id']): s for i, s in enumerate(refs)}
        coco = CocoVqa(gts, refs, None)
        results = coco.evaluate()
        for k, v in results.items():
            if k == 'accuracy':
                print(k)
                print(v)

    # localization

    # gts = [s for s in gts if 'original_detection' in s]
    #
    # for size in ['small', 'base', 'large', 'xl']:
    #     print(size)
    #     refs = json.load(open('../temp/cococon_task_outputs_%s.json' % size))
    #     refs_loc = {str(i+1): s for i, s in enumerate(refs) if 'predict_bboxes' in s}
    #     boxes = {str(i+1): {'boxes': np.array([np.array(bbox) for bbox in s['predict_bboxes']]), 'relevance': np.array([1.0]*len(s['predict_bboxes']))} for i, s in enumerate(refs) if 'predict_bboxes' in s}
    #     coco = CocoDetection(gts, refs_loc, boxes)
    #     results = coco.evaluate()
    #     for k, v in results.items():
    #         print(k)
    #         print(v)
