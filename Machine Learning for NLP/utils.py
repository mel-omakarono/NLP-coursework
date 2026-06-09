from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def most_common_words(a_list, num_words=20):
    """Returns the most frequent words from a list of dictionaries containing 'token' as key.

    Parameters:
        a_list (list): List of dictionaries containing 'token'.
        num_words (int): Number of most frequent words to return (default is 20).

    Returns:
        List of tuples: Most common words with their frequencies."""
    words_list=[]
    for dic in a_list:
        word = dic['token']
        words_list.append(word)
    counts_words = Counter(words_list).most_common(num_words)
    return counts_words

def most_common_pos(a_list, num_words=20):
    """Returns the most frequent POS tags from a list of dictionaries containing 'pos' as key.

    Parameters:
        a_list (list): List of dictionaries containing 'pos'.
        num_words (int): Number of most frequent POS tags to return (default is 20).

    Returns:
        List of tuples: Most common POS tags with their frequencies."""
    words_list=[]
    for dic in a_list:
        word = dic['pos']
        words_list.append(word)
    counts_pos = Counter(words_list).most_common(num_words)
    return counts_pos

def analyze_token_length(a_list):
    """Analyzes token lengths in a list of dictionaries and returns average, max, and min token length.

    Parameters:
        a_list (list): List of dictionaries containing 'token'.

    Returns:
        tuple: Average, max, and min token lengths."""
    lengths = [len(dic['token']) for dic in a_list]
    lengths = [len(dic.get('token', '')) for dic in a_list]
    avg_length = sum(lengths) / len(lengths)
    values = [avg_length, max(lengths), min(lengths)]
    return avg_length, max(lengths), min(lengths)

def capitalization_patterns(a_list):
    """Analyzes and returns the distribution of capitalization patterns in a list of dictionaries containing 'token'.

    Parameters:
        a_list (list): List of dictionaries containing 'token'.

    Returns:
        dict: Counts of each capitalization pattern (uppercase, lowercase, title case, other)."""
    capitalization = {"uppercase": 0, "lowercase": 0, "first_letter_capitalization": 0, "other": 0}
    for dic in a_list:
        token = dic['token']
        if token.isupper():
            capitalization["uppercase"] += 1
        elif token.islower():
            capitalization["lowercase"] += 1
        elif token.istitle():
            capitalization["first_letter_capitalization"] += 1
        else:
            capitalization["other"] += 1

    labels = capitalization.keys()
    sizes = capitalization.values()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['gold', 'lightblue', 'lightcoral', 'lavender'])
    plt.title('Capitalization Patterns')
    plt.axis('equal') 
    plt.show()
    return capitalization

def extract_less_features(inputfile):
    """Takes as input the path to a conll file and returns a list of NER labels."""   
    labels = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                label = components[1]
                labels.append(label)
    return labels

def find_prec_recall_fscore_matrix(onefile, twofiles):
    '''Computes precision, recall, F1-score, and confusion matrix for gold and predicted labels from two files.

    Parameters:
        onefile (str): Path to the gold label file.
        twofiles (str): Path to the predicted label file.

    Returns:
        str: Classification report as a string.'''
    gold_labels = extract_less_features(onefile)
    pred_labels = extract_less_features(twofiles)
    unique_gold = set(gold_labels)
    unique_pred = set(pred_labels)  
    unique_labels = sorted(unique_gold.union(unique_pred))
    precision = precision_score(gold_labels, pred_labels, average='weighted')
    recall = recall_score(gold_labels, pred_labels, average='weighted')
    f1 = f1_score(gold_labels, pred_labels, average='weighted')
    confusion = confusion_matrix(gold_labels, pred_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion,
                                     display_labels=unique_labels)
    display.plot(cmap=plt.cm.magma)
    plt.title("Sample Confusion Matrix")
    plt.show()
    report = classification_report(gold_labels, pred_labels)
    return report

def metrics_and_matrix(file):
    '''Computes classification report and confusion matrix from a CoNLL file with gold and predicted labels in the last columns.

    Parameters:
        file (str): Path to the CoNLL file.

    Returns:
        str: Classification report as a string.
        Shows confusion matrix.'''
    new_gold_labels = []
    new_pred_labels = []
    with open(file, 'r') as infile:
        for line in infile:
            if line.strip():  
                parts = line.strip().split()
                new_gold_labels.append(parts[-2])  
                new_pred_labels.append(parts[-1])
    
    labels=['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
    new_conll_report = classification_report(new_gold_labels, new_pred_labels, labels=labels, digits=3)
    
    confusion = confusion_matrix(new_gold_labels, new_pred_labels, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion,
                                         display_labels=labels)
    display.plot(cmap=plt.cm.magma_r)
    plt.title("Confusion Matrix")
    plt.show()
    return new_conll_report
