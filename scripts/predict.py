#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kaivalya mannam
"""
import sys
import os
import glob
import numpy as np
import copy
from collections import Counter
from flair.models import SequenceTagger
from flair.data import Token, Sentence
from unify_helper import readTextFile
from unify_helper import makeSentences_for_predict
import time

elapsed_times = None

def main():

    global elapsed_times
    elapsed_times = []
    
    model_files = ["model/final-model.pt"]    
    for model_file in model_files:
        if not os.path.exists(model_file):
            print("cannot find model_file -" + model_file)
            sys.exit(1)
    
    dataset_number = 1
    
    gold_dir = "data/test"
    if not os.path.exists(gold_dir):
        print("Gold dir {} doesnt exist".format(gold_dir))
        sys.exit(1)
        
    dest_dir = "output"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    
    taggers = []
    for i, model_file in enumerate(model_files):
        elapsed_times.append({'freqs':{}, 'times': {}})
        print ("Loading model_file: " + model_file)
        seqtagger = SequenceTagger.load(model_file)
        taggers.append({'model_id': i, 'model': seqtagger})
    
    predict(taggers, gold_dir, dest_dir) 

    with open(dest_dir+'/times.txt', "w") as f:
        f.write("model_id\tword_len\tfrequency\tcumulative_ms\n")
        for i in range(0, len(model_files)):
            obj = elapsed_times[i]
            freqs_inst = obj['freqs']
            times_inst = obj['times']
            for key in freqs_inst:
                freq = freqs_inst[key]
                elapsed = times_inst[key]
                f.write("{}\t{}\t{}\t{}\n".format(i, key, freq, elapsed))
        
# returns entity suited to write in output .ann file
# also updates sent array with the tokens suited for our processing (target/secondary_target)            
def make_entity(sent, all_entities, sp, enable_map, model_id):
    
    # sp.tokens[i].idx directly indexes into sent array
    # sent['words'] gives us the original words
    # sent['starts] gives us the original start indices            
    span_text = ""
    index = sp.tokens[0].idx 
    start = sp.tokens[0].start_position
    end = sp.tokens[0].start_position
    line_num = int(sent['line_num'][index])
    existing = [x for x in all_entities if x['start'] == start ]
    if len(existing)>0:
        print("WARNING - attempting to add duplicate to all entities")
    for l in range(0, len(sp.tokens)):
        if l == 0:
            span_text = sent['words'][index]
            end = end + len(span_text)            
        else:   
            # add space only if required as (1) will be tokenized into ( 1 ) and
            # we shouldn't be putting space unless 'starts' tell us to do so
            # in case there is a gap between where we should be and where we are 
            # do some padding 
            while end < int(sent['starts'][index]):
                span_text = span_text + " "
                end = end + 1
            span_text = span_text + sent['words'][index]
            end = end + len(sent['words'][index])
        index = index + 1
        
    e = {'text': span_text, 'start': start, 'end': end, 'type': sp.tag, 'orig_span': sp, 'line': line_num, 'model_ids':[model_id], 'scores': [sp.score]}
    e['name']='T' + str(len(all_entities)+1)
    all_entities.append(e)

def make_entities(sent, all_entities, spans, enable_map, model_id):
    for sp in spans:   
        existing = [x for x in all_entities if x['start'] == sp.start_pos ]        
        if len(existing)>0:
            if model_id not in existing[0]['model_ids']:
                existing[0]['model_ids'].append(model_id)
                existing[0]['scores'].append(sp.score)        
        if len(existing)==0:
            make_entity(sent, all_entities, sp, enable_map, model_id)
        elif existing[0]['type']!=sp.tag:
            # Allow ADE to override Reason and Duration to override Frequency
            if sp.tag in ('ADE', 'Duration'):
                print("Existing: {} New {}".format(existing[0]['type'], sp.tag))
                existing[0]['type']=sp.tag    
                existing[0]['orig_span']=sp
            elif sp.tag in ("Dosage", "Frequency") and existing[0]['type'] in ("Dosage", "Frequency") and sp.score>existing[0]['scores'][0]:
                print("Existing: {} New {} - {}".format(existing[0]['type'], sp.tag, sp.tokens[0]))
                existing[0]['type']=sp.tag    
                existing[0]['orig_span']=sp
            elif sp.tag in ("Form", "Route") and existing[0]['type'] in ("Form", "Route") and sp.score>existing[0]['scores'][0]:
                print("Existing: {} New {} - {}".format(existing[0]['type'], sp.tag, sp.tokens[0]))
                existing[0]['type']=sp.tag    
                existing[0]['orig_span']=sp
            elif sp.tag in ("Strength", "Dosage") and existing[0]['type'] in ("Strength", "Dosage") and sp.score>existing[0]['scores'][0]:
                print("Existing: {} New {} - {}".format(existing[0]['type'], sp.tag, sp.tokens[0]))
                existing[0]['type']=sp.tag    
                existing[0]['orig_span']=sp
            elif sp.tag in ('Drug', 'Form') and existing[0]['type'] in ('Drug', 'Form') and sp.score>existing[0]['scores'][0]:
                print("Existing: {} New {} - {}".format(existing[0]['type'], sp.tag, sp.tokens[0]))
                existing[0]['type']=sp.tag    
                existing[0]['orig_span']=sp
            else:
                print("WARNING: Existing: {} {} New {} {} Start {}- Ignored".format(existing[0]['type'], existing[0]['scores'][0], sp.tag, sp.score, sp.start_pos))

def predict_sentence_entities(tagger, sent, all_entities, phase=1):

    #print("Processing {}".format(sent['words'][0]))
    global elapsed_times
    newsent = Sentence()     
    for i in range(0, len(sent['normwords'])):
        tok = sent['normwords'][i]
        token = Token(tok, i, None, start_position=int(sent['starts'][i]))
        newsent.add_token(token)
    
    seqtagger = tagger['model']
    model_id = tagger['model_id']
    
    start = time.time()
    seqtagger.predict(newsent)
    end = time.time()
    words = len(sent['normwords'])
    elapsed = np.round((end-start)*1000,0) 
    if elapsed<0:
        elapsed = 0
    obj = elapsed_times[model_id]
    if obj['freqs'].get(words)!=None:
        obj['freqs'][words] = obj['freqs'][words]+1
        obj['times'][words] = obj['times'][words] + elapsed
    else:
        obj['freqs'][words] = 1
        obj['times'][words] = elapsed
    ner_spans = newsent.get_spans("ner")
    make_entities(sent, all_entities, ner_spans, False, model_id)
    return
    
def write_entities(gold_folder, pred_folder, text_file, ann_file, entities): #, tag_counter):
    
    #entities.sort(key=lambda x: x['start'])
        
    with open(pred_folder + '/entities.txt', 'a') as f:
        for e in entities:                
                f.write("{} {} {} {}".format(text_file, e['type'], e['text'], e['start']))
                for model_id in e['model_ids']:
                    f.write(" {}".format(str(model_id)))
                for score in e['scores']:
                    f.write(" {}".format(str(score)))
                f.write("\n")
    
    # write all output
    with open(ann_file.replace(gold_folder, pred_folder), "wt") as f:
        for i, e in enumerate(entities):
            #tag_counter.update([e['type']])
            f.write("T{}\t{} {} {}\t{}\n".format(i+1, e['type'], e['start'], e['end'], e['text']))            
  
        
# for each txt file in gold_folder and generates .ann file in pred_folder
def predict(taggers, gold_folder, pred_folder):
    
    text_files = glob.glob(gold_folder + "/*.txt")
    
    # for STL, we will generate Sentence result, lookahead by 1 result, and their union    
    for i in range(1, 4):
        try:
            # create directory
            os.makedirs(pred_folder + "/"+str(i))
        except: 
            # if directory exists truncate entities file
            with open(pred_folder + "/"+str(i)+"/entities.txt", 'w') as f:
                print("truncated entities.txt")        
    
    #tag_counter = Counter()
    for i in range(0, len(text_files)):

        print("Predict: "+text_files[i])

        ann_file = text_files[i].replace(".txt", ".ann")

        # read the txt file
        text_info = readTextFile(text_files[i])

        # convert text info to sentences as per our usual logic (one sentence is a prediction entity)
        sentences1 = makeSentences_for_predict(text_info, Counter(), Counter(), None)

        # Pass 1 - sentence level predictions, every entity is included
        entities1 = []        
        for sent in sentences1:
            if not sent: continue
            newsent = copy.deepcopy(sent)
            for tagger in taggers:
                predict_sentence_entities(tagger, newsent, entities1, 1)
        entities1.sort(key=lambda x: x['start'])
        write_entities(gold_folder, pred_folder + "/1", text_files[i], ann_file, entities1)
        
        if len(taggers)>1:
            continue
        
        # Pass 2 - lookahead 1 predictions (all)
        entities2 = []                 
        for si, sent in enumerate(sentences1):
            if not sent: continue
            newsent = copy.deepcopy(sent)
            if si+1<len(sentences1):
                for key in ["words", "normwords", "starts", "line_num", "word_index", "targets", "secondary"]:
                    newsent[key] = newsent[key] + copy.deepcopy(sentences1[si+1][key])
            for tagger in taggers:
                predict_sentence_entities(tagger, newsent, entities2, 2)
        entities2.sort(key=lambda x: x['start'])
        write_entities(gold_folder, pred_folder + "/2", text_files[i], ann_file, entities2)
        
        # for STL models, we will generate a third set of predictions by 
        # augmenting ADE entities discovered by look-ahead-1 strategy
        entities = entities1
        newentities2 = entities2 
        for e in newentities2:
            if e['type'] not in ('ADE', 'Reason'): 
                continue
            existing = [x for x in entities if x['start'] == e['start'] ]
            if len(existing)==0:
                # we can add ADE / Reason if they are new.
                entities.append(e)
            elif existing[0]['type']!=e['type'] and e['type']=='ADE':
                # Allow ADE augmentation only, skip conflicting Reason
                print("{} Existing: {} New {}".format(3, existing[0]['type'], e['type']))
                existing[0]['type']=e['type']
                existing[0]['orig_span']=e['orig_span']
        entities.sort(key=lambda x: x['start'])
        write_entities(gold_folder, pred_folder + "/"+str(3), text_files[i], ann_file, entities)
                
if __name__ == "__main__":
    main()


