1. unify.py

example: python3 -m unify

this takes the data/train files and generates IOB compatible output file.

2. trainer.py

example: python3 -m trainer ep

executes model training with specified embedding using the data/dataset1_train.txt and data/dataset1_test.txt as training/test files (for demonstration)

writes the final model to model/ sub-directory.

3. rel_predict_new.py

example: python3 -m rel_predict_new

evaluates the model on the test set (from data/test) and writes to output/ directory.

4. track2-evaluate-ver4.py

example: python3 -m track2-evaluate-ver4 data/test output

compares the test directory with output directory and outputs the evaluation score.





