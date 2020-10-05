#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kaivalya mannam
"""

import glob
from collections import Counter
from preprocess_helper import readTextFile, readAnnFile, readEntities
from preprocess_helper import makeSentences, isNumber

# unify

def main():
    unify("data/train", "data/dataset1_train.txt")
    unify("data/test", "data/dataset1_test.txt")

# Merges data in txt and ann files to provide labels for words

def unify(folder, t_filename):

    text_files = glob.glob(folder + "/*.txt")

    t_output = open(t_filename, "w")    
        
    orig_tok_counter = Counter()
    orig_rel_counter = Counter()
    new_tok_counter = Counter()
    sent_len_counter = Counter()

    for i in range(0, len(text_files)):

        #print("Processing file -"+text_files[i])

        ann_file = text_files[i].replace(".txt", ".ann")

        # read the txt file
        text_info = readTextFile(text_files[i])

        # get the annotated lines from the file
        t_lines, t_stats, r_lines, r_stats = readAnnFile(ann_file)

        orig_tok_counter.update(t_stats)
        orig_rel_counter.update(r_stats)

        # read the ann file and update text_info
        text_info, entity_dict = readEntities(t_lines, text_info)

        # convert text_info to sentences
        sentences = makeSentences(
            text_info, new_tok_counter, sent_len_counter)

        # write tokens
        writeSeqFile(sentences, t_output, text_files[i])
       
    t_output.close()
    print("Original Statistics:\n\tTokens:\n")
    print(orig_tok_counter)
    print("\tRels:\n")
    print(orig_rel_counter)
    print("New Statistics:\n\tTokens:")
    print(new_tok_counter)
  
    #print("Sentence Lengths:\n")
    #print(sent_len_counter.most_common(50))

def writeSeqFile(sentences, output_file, text_file):

    # for each sentence
    for sent in sentences:

        # write a line for each word
        for i in range(0, len(sent['words'])):

            # get the word and check if its a number
            word = sent['words'][i]
            origword = word
            numeric, newword = isNumber(word)
            if numeric:
                word = newword
            # write the line
            output_file.write(" ".join([text_file, sent['line_num'][i], sent['word_index'][i], sent['seq'][i], sent['starts'][i], str(
                int(sent['starts'][i]) + len(origword)), origword, word, sent['targets'][i]]))
            output_file.write('\n')
        output_file.write('\n')
            

if __name__ == "__main__":
    main()
