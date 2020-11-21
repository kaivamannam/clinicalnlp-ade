This repository contains the preprocessing, training, and prediction evaluation scripts for the following paper.

Paper:              https://www.aclweb.org/anthology/2020.clinicalnlp-1.6/

Video Presentation: https://slideslive.com/38939820

Citation:

                    @inproceedings{narayanan-etal-2020-evaluation,
                        title = "Evaluation of Transfer Learning for Adverse Drug Event ({ADE}) and Medication Entity Extraction",
                        author = "Narayanan, Sankaran  and
                          Mannam, Kaivalya  and
                          Rajan, Sreeranga P  and
                          Rangan, P Venkat",
                        booktitle = "Proceedings of the 3rd Clinical Natural Language Processing Workshop",
                        month = nov,
                        year = "2020",
                        address = "Online",
                        publisher = "Association for Computational Linguistics",
                        url = "https://www.aclweb.org/anthology/2020.clinicalnlp-1.6",
                        pages = "55--64"
                    }




You are welcome to use the code for your projects or research. If you find it useful,
we request you to cite us. If you find any issues with the scripts, let us know.

===========================================

Dataset: 2018 n2c2 ADE and Medication Extraction Dataset, available from 
https://portal.dbmi.hms.harvard.edu/.  

Training procedure:
Use the preprocess.py script to convert the dataset into CoNLL2000 BIO format which is then used to train via the train.py script. On the final model, you can use predict.py to generate predictions back in the dataset format. The track2-evaluate-ver4.py (in misc/) directory can be used to generate the 'Relaxed F1' performance scores.

Scripts were tested on: Python 3.6, Flair 0.4.5, and allennlp 0.9.0. Torch 1.5.1 is recommended.

For Flair-PubMed we attempted language model fine-tuning using the procedure in 
https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md

The resultant .pt files are available from 
https://drive.google.com/open?id=1occF-vL4f8KT1kmApJXSXYFVtARLhcAW

For ClinicalBERT and BioBERT, we use their pytorch-compatible versions. Please them in the script/ directory and adjust the paths as needed in trainer.py.



