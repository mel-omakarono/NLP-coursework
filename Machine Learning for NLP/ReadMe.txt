Hello professors,

To run my models, you need to take the following steps:

A. 'final_py_file.py' file:

1. save it anywhere, preferably close to where the conll files and the word embeddings model are so that the paths are not very long

2. It requires the path to a train file, the path to an input file (dev or test), the path to a new file that will be created and contain the results, and the model name. 
Options for model_name parameter are: "logreg", "NB", "SVM"
Example commands look like this:

[INSERT PATH TO WHERE YOU SAVED IT HERE] python final_py_file.py "../data/conll2003/conll2003.train.conll" "../data/conll2003/conll2003.dev.conll" "finaltest.conll" --model_name "logreg"

[INSERT PATH TO WHERE YOU SAVED IT HERE] python final_py_file.py "../data/conll2003/conll2003.train.conll" "../data/conll2003/conll2003.dev.conll" "finaltest.conll" --model_name "NB"

[INSERT PATH TO WHERE YOU SAVED IT HERE] python final_py_file.py "../data/conll2003/conll2003.train.conll" "../data/conll2003/conll2003.dev.conll" "finaltest.conll" --model_name "SVM"

If no model is specified, the default is logistic regression.

3. In the case of word embeddings, my classifier only runs with SVM. It does not matter if you specify a different name. Here, you will have to add two more parameters, embedded, and language model path.
An example command looks like this:

[INSERT PATH TO WHERE YOU SAVED IT HERE] python final_py_file.py "../data/conll2003/conll2003.train.conll" "../data/conll2003/conll2003.dev.conll" "finaltest.conll" --model_name "logreg" --embedded --language_model_path "../models/GoogleNews-vectors-negative300.bin.gz"

The "embedded" parameter simply needs to be mentioned to be true, and the "language_model_path" parameter requires the path to the language model after its mention.

Please note: in the case of word embeddings, two additional files will automatically be created "all_train.conll" and "all_input.conll". They are files with extra columns that contain all my features, so that the code for word embeddings can run. 




B. basic system notebook file

1. The notebook needs to be placed in its original folder in the original structure you provided us with (assignment1 under code). 

2. utils.py and ner_machine_learning.py are both used, so they are attached and should be placed in the same folder (assignment1).

3. If the conll files are under "data" in the same structure, then you do not need to change the paths. Otherwise, please change the paths.

4. If the language model for the word embeddings is under "models" in the same structure, then you do not need to change the path. Otherwise, please change the path. 

5. For my convenience, I created a separate cell for a normal SVM, as when the "create classifier" function is run for SVM does the hyperparameter tuning with Grid search, therefore it takes a while. Please take this into account when you are running my notebook. 

6. In the evaluation section, we had to create some files with made-up labels and create a confusion matrix. I attached a folder called "fake files", which should be in the same folder as the basic system notebook, otherwise this part of the code will not work. 



Please note that argparse is used in the main function. Other libraries required include: 
numpy, csv, gensim, pandas, sklearn, matplotlib, collections