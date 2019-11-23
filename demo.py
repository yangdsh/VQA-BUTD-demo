#!/usr/bin/env python

from fabric.api import (env, local, task, lcd)
from six.moves import urllib
import argparse
import json
import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='VQA demo')
    parser.add_argument('-u', dest='url', help='image url',
                        default='', type=str)
    parser.add_argument('-q', dest='question',
                        help='question',
                        default='How are you?', type=str)
    parser.add_argument('-n', dest='num_ans',
                        help='number of answers',
                        default=1, type=int)

    args = parser.parse_args()
    return args
                      
     
if __name__ == '__main__':

    args = parse_args()

    # when a new image is used
    if args.url is not '':
        # download image
        urllib.request.urlretrieve(args.url, 'bottom-up-attention/data/demo/example2.jpg')
        
        # clear the cache
        cmd = 'rm bottom-up-attention/res.tsv.0'
        if os.path.exists('bottom-up-attention/res.tsv.0'):
            local(cmd)
        
        # generate image feature
        with lcd('./bottom-up-attention/'):
            cmd = 'python2 tools/generate_tsv.py'
            local(cmd)
        
        # process the image feature
        with lcd('./bottom-up-attention-vqa/'):
            cmd = 'python2 tools/detection_features_converter.py'
            local(cmd)
    
    # store the question
    out = {"questions": [{"question_id": 0, "question": args.question}]}
    with open('question.json', 'w') as f:
        json.dump(out, f)
    
    # run vqa model
    with lcd('./bottom-up-attention-vqa/'):
        local('python2 main.py --num_ans {} --demo'.format(args.num_ans))
    
    # visualize attention
    with lcd('./bottom-up-attention/tools/'):
        local('ipython2 draw_attentions.py')
                  
