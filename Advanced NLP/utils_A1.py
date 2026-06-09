import pandas as pd
import spacy
from spacy.tokens import Doc
import en_core_web_md
import benepar
# uncomment if using benepar_en3 for the first time
# benepar.download('benepar_en3')
nlp_simple = spacy.load("en_core_web_md")
nlp_benepar = spacy.load("en_core_web_md")
nlp_benepar.add_pipe('benepar', config={'model': 'benepar_en3'})
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import nltk
from collections import Counter
from tqdm import tqdm
import warnings
# Suppress PyTorch warnings that cause multiple progress bars
warnings.simplefilter("ignore")
import networkx as nx
from benepar.spacy_plugin import BeneparComponent
from nltk.tree import Tree
import pickle

def get_constituency_path(sentence_tokens, sentence_predicate):
    """
    Processes a sentence with tokens and a corresponding predicate list to create constituency 
    paths for each token, marking the path from the root to the token in the constituency parse tree.
    Additionally, it marks the predicate with a specific label.

    Parameters:
        sentence_tokens (list of str): List of tokens in the sentence.
        sentence_predicate (list of int): List that marks the predicate (1 for predicate, 0 otherwise).
        
    Returns:
        list of dicts: A list of dictionaries, each containing the 'constituency_path' key that holds the path 
                       to the token's constituency in the parse tree. If no predicate is found, "UNK" is returned.
    """
    punctuation_map = {
        '[': 'LSB',  # Left Square Bracket
        ']': 'RSB',  # Right Square Bracket
        '(': 'LRB',  # Left Round Bracket
        ')': 'RRB',  # Right Round Bracket
        '{': 'LCB',  # Left Curly Bracket
        '}': 'RCB'   # Right Curly Bracket
    }
    # if sentence_tokens is a list, processes the tokens using the benepar NLP pipeline (better for constituency)
    if isinstance(sentence_tokens, list):
        spacy_doc = Doc(nlp_benepar.vocab, words=sentence_tokens)
        processed_doc = nlp_benepar(spacy_doc)

    token_info = []  # stores the constituency paths for each token

    for sent in processed_doc.sents:
        tree = Tree.fromstring(sent._.parse_string)  # creates a constituency tree from the parsed sentence
        tree_leaves = tree.leaves()  # extracts the leaves of the tree (tokens)

        try:
            # finds the index of the predicate in the sentence_predicate list
            predicate_index = sentence_predicate.index(1)
        except ValueError:
            # returns "UNK" for all tokens if no predicate is found (for safety)
            return [{'constituency_path': 'UNK'} for _ in sentence_tokens]

        predicate_tree_position = None  # stores the tree position of the predicate
        predicate_path = []  # stores the full path to the predicate

        for i, (token, predicate) in enumerate(zip(sent, sentence_predicate)):
            token_text = punctuation_map.get(token.text, token.text)  # maps punctuation 
            try:
                # finds the index of the token in the tree's leaves
                leaf_index = tree_leaves.index(token_text)
                tree_position = tree.leaf_treeposition(leaf_index)  # gets the position of the token in the tree

                constituency_path = []  # stores the constituency path for this token
                current_tree = tree  # starts at the root of the tree

                # builds the path from root to the token
                for position in tree_position:
                    if isinstance(current_tree, Tree):
                        constituency_path.append(current_tree.label())  # Add the tree label to the path
                        current_tree = current_tree[position]  # Move to the next tree node

                # If the predicate is found, store its tree position and path
                if predicate == 1:
                    predicate_tree_position = tree_position
                    predicate_path = constituency_path  # Store the full path to the predicate

                # Truncate the path for tokens before the predicate (only keep common prefix)
                if i < predicate_index and predicate_tree_position:
                    predicate_level = len(predicate_tree_position)
                    constituency_path = constituency_path[:predicate_level]

                # truncates the path for tokens after the predicate
                if i > predicate_index and predicate_path:
                    common_prefix_len = len([x for x, y in zip(constituency_path, predicate_path) if x == y])
                    constituency_path = constituency_path[common_prefix_len:][::-1]  # Reverse the differing part
                
                # marks the predicate explicitly in the constituency path
                if predicate == 1:
                    constituency_path.append("PREDICATE")

                # joins the path as string
                new_label = '-'.join(constituency_path)

            except ValueError:
                # accounting for errors such as missing values
                new_label = "UNK"

            token_info.append({'constituency_path': new_label})

    return token_info




def find_arg_not_0_and_1(sentence_token):
    """
    Processes a list of sentence tokens and writes a feature based on token types, 
    POS tags, and NER labels. It constructs a feature list for each token in the sentence 
    based on its previous tokens.

    Parameters:
        sentence_token (list of str): A list of tokens in the sentence to process.
        
    Returns:
        list of dicts: A list of dictionaries, each containing the 'prep_ner' key. This key stores 
                       a label for the token, based on its NER type or POS tag and its context.
    """
    
    features_list = []  # stores feature dicts for each token in the sentence

    # checks if sentence_token is a list
    if isinstance(sentence_token, list):
        # processes with standard nlp pipeline (without benepar) for faster computation
        spacy_doc = Doc(nlp_simple.vocab, words=sentence_token)
        processed_doc = nlp_simple(spacy_doc)

        for sent in processed_doc.sents: # sentence level
            tokens = list(sent)  
            
            # token level
            for i, token in enumerate(tokens):
                # feature for current token
                features_dict = {
                    'prep_ner': ''  
                }
                features_list.append(features_dict)

                # if ner
                if token.ent_type_:
                    # if first token in the sentence, just adds the ner label
                    if i == 0:
                        final_label = f'{token.ent_type_}'
                    else:
                        # if the previous token has an entity type or is a functional word (ADP, DET, VERB, AUX)
                        if tokens[i-1].ent_type_ or tokens[i-1].pos_ in {'ADP','DET','VERB','AUX'}:
                            final_label = f'{tokens[i-1].lemma_}|{token.ent_type_}'
                        # if the token is preceded by a preposition (ADP) and determiner (DET)
                        elif tokens[i-2].pos_ =='ADP' and tokens[i-1].pos_ == 'DET':
                            final_label = f'{tokens[i-2].lemma_}|{tokens[i-1].lemma_}|{token.ent_type_}'
                        else:
                            final_label = f'{token.ent_type_}'
                
                # if the token is a pronoun or noun
                elif token.pos_ in {'PRON','NOUN'}:
                    # if first token in sentence
                    if i == 0:
                        final_label = f'{token.pos_}'
                    else:
                        # if the previous token is a functional word (ADP, DET, VERB, AUX)
                        if tokens[i-1].pos_ in {'ADP','DET','VERB','AUX'}:
                            final_label = f"{tokens[i-1].lemma_}|{token.pos_}"
                        # if the token is preceded by a preposition (ADP) and determiner (DET)
                        elif tokens[i-2].pos_ =='ADP' and tokens[i-1].pos_ == 'DET':
                            final_label = f'{tokens[i-2].lemma_}|{tokens[i-1].lemma_}|{token.pos_}'
                        else:
                            final_label = f'{token.pos_}'
                else:
                    # using simple POS tags for tokens that are not in the above categories
                    final_label = f'{token.pos_}'
                
                if final_label:
                    features_list[i]['prep_ner'] = final_label

    return features_list



def build_dependency_graph(processed_doc):
    """
    Constructs a dependency graph from a processed sentence.

    Takes a processed spaCy doc object and creates a graph where each token 
    is represented as a node, and edges show syntactic dependencies between tokens. The graph 
    allows bidirectional traversal and each dependency relation is stored in both directions.

    Parameters:
        processed_doc: A spaCy-processed sentence containing dependency annotations.

    Returns:
        tuple: 
            - nx.Graph: undirected graph where nodes are token text (lowercased) 
              and edges represent dependency relations.
            - dict: mapping of (token 1, token 2) → direction ("↑" for child → parent, 
              "↓" for parent → child).
            - dict: mapping of (token 1, token 2) → dependency label.
    """
    edges = []
    dep_directions = {}
    dep_labels = {}

    for token in processed_doc:
        for child in token.children:
            edges.append((child.text.lower(), token.text.lower()))  # child → parent
            edges.append((token.text.lower(), child.text.lower()))  # parent → child
            dep_directions[(child.text.lower(), token.text.lower())] = "↑"
            dep_directions[(token.text.lower(), child.text.lower())] = "↓"
            dep_labels[(child.text.lower(), token.text.lower())] = child.dep_
            dep_labels[(token.text.lower(), child.text.lower())] = child.dep_

    return nx.Graph(edges), dep_directions, dep_labels


def find_dep_path(graph, dep_directions, dep_labels, word, predicate):
    """
    Finds the shortest dependency path between a given word and a predicate 
    in a dependency graph (with directions).

    Parameters:
        graph (nx.Graph): dependency graph where nodes are words (lowercased).
        dep_directions (dict): dict mapping (word 1, word 2) → direction 
                               ("↑" for child → parent, "↓" for parent → child).
        dep_labels (dict): dict mapping (word 1, word 2) → dependency label.
        word (str): starting word in the dependency tree.
        predicate (str): predicate word in the dependency tree.

    Returns:
        str: string representing the shortest dependency path 
             with dependency labels and direction arrows (e.g., "nsubj ↑ prep ↓").
             Returns "No Path" if no path exists between the given words.
    """
    try: # shortest dependency path between the given word and predicate
        dep_path = nx.shortest_path(graph, source=word.lower(), target=predicate.lower())
        # dependency labels and directional arrows
        arrow_path = [
            f"{dep_labels.get((dep_path[i], dep_path[i+1]), '?')} {dep_directions.get((dep_path[i], dep_path[i+1]), '?')}"
            for i in range(len(dep_path) - 1)
        ] 
        return " ".join(arrow_path)
    except (nx.NetworkXNoPath, ValueError, nx.NodeNotFound):
        return "No Path"



def extract_features_from_dataframe(df):
    """
    Extracts features for each token in the df.

    Parameters:
    df (DataFrame): input df containing sentences and tokens.

    Returns:
    list: A list of dictionaries containing features for each token.
    """
    df_features = []
    total_tokens = len(df)

    df['chosen_predicate'] = df['chosen_predicate'].astype(int)

    # progress bar
    pbar = tqdm(total=total_tokens, desc="Processing Tokens", unit="token", leave=True, dynamic_ncols=True, position=0)
    
    for sentence_num, sentence in df.groupby('sentence_num', sort=False):
        sentence_feats = []
        sentence_tokens = sentence['token'].astype(str).tolist()
        sentence_predicate = sentence['chosen_predicate'].tolist()

        # processing sentence with spaCy
        spacy_doc = Doc(nlp_simple.vocab, words=sentence_tokens)
        processed_doc = nlp_simple(spacy_doc)

        # building dependency graph once per sentence
        graph, dep_directions, dep_labels = build_dependency_graph(processed_doc)

        predicate_token = None
        predicate_word = None
        predicate_lemma = None

        for idx, (word, pred) in enumerate(zip(sentence['token'], sentence['chosen_predicate'])):
            if pred == 1:
                predicate_token = processed_doc[idx]
                predicate_word = word
                predicate_lemma = predicate_token.lemma_    # finding lemma
                break  

        arg_features = find_arg_not_0_and_1(sentence_tokens)    # checking if token is ner
        constituency_features = get_constituency_path(sentence_tokens, sentence_predicate)    # finding constituency path

        for token_idx, token in enumerate(processed_doc):
            # using graph made at sentence level for dependency path
            dep_path = find_dep_path(graph, dep_directions, dep_labels, token.text, predicate_word) if predicate_word else "No Path"

            arg_feature_dict = arg_features[token_idx] if token_idx < len(arg_features) else {'prep_ner': ''}
            constituency_feature_dict = constituency_features[token_idx] if token_idx < len(constituency_features) else {'constituency_path': ''}

            feature_dict = {
                'dep_path_plus_lemma': f'{dep_path} - {predicate_lemma}' if predicate_lemma else dep_path,
                'is_it_ner': arg_feature_dict['prep_ner'],
                'constituency_path': constituency_feature_dict['constituency_path']
            }
            sentence_feats.append(feature_dict)

        df_features.extend(sentence_feats)
        pbar.update(len(sentence))

    pbar.close()
    
    return df_features
