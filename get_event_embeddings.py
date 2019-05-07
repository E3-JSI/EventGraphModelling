import pandas as pd 
from gensim.models import KeyedVectors,fasttext
from gensim.test.utils import datapath
import pdb 
import string 
import numpy as np 
import json 

def get_concepts(concept_list):
    """
    Get all concepts from the events data, 
    
    :param concept_list: dictionary of concept properties in the EventRegistry format
    
    :return: tuple of concept & weight
    """
    concepts = [(conc["label"]["eng"],conc["score"]) for conc in concept_list]
    return concepts 

def get_event_embedding(model,concepts):
    """ 
    Get event emebeddings from concepts by formula
    e_i = sum_j w_ij * word2vec(c_ij) 

    :param model: gensim word2vec model
    :param concepts: list of tuples with string concept at position 0 and wgt at position 1

    :return: np.array embedding event in the model's word2vec space
    """
    res = np.zeros(model["test"].shape[0])
    for conc in concepts:
        word = conc[0]
        if word in model.vocab:
            word_vec= model[word]
        else:
            words = word.translate(str.maketrans("","",string.punctuation)) # remove punctuation
            words = word.split() # split into separate words
            word_vec = np.average([model[wrd] for wrd in words if wrd in model.vocab],0)
            if type(word_vec) is np.float64: continue # no embedding found

        res += conc[1]/100 * word_vec

    # In ideal world
    #res = sum([conc[1]/100 * model[conc[0]] for conc in concepts])

    return res 

# Load settings file with all the paths
settings = json.load(open("settings.json","r"))

# Load news events
print("Loading news events")
events_path = settings["events_path"]
events = pd.read_json(events_path,lines=True)
events["concepts"] = events["concepts"].apply(lambda x: get_concepts(x))

# import word2vec bin
print("Loading word2vec model")
word2vec_google = settings["word2vec_google"]
word2vec_fb =     settings["word2vec_fb"]
word2vec_wiki1 =  settings["word2vec_wiki1"]
word2vec_wiki2 =  settings["word2vec_wiki2"]

# Load the pretrained model
model = KeyedVectors.load_word2vec_format(word2vec_google, binary = True)
#model = fasttext.load_facebook_model(datapath(word2vec_fb))
model.init_sims(replace = True) # normalizes the vectors in the word2vec class

print("Getting event emebeddings")
events["concepts_embd"] = events["concepts"].apply(lambda x: get_event_embedding(model,x))
# Transform dates to unix
events["eventDate"] = pd.to_datetime(events["eventDate"]).apply(lambda x: x.timestamp())

# Combine embd with date
embd = pd.DataFrame([np.append(embd,date) for embd,date in events[["concepts_embd","eventDate"]].values])

# Saving the file
pdb.set_trace()
out_path = settings["out_path"]+"concepts_date_embd.csv"
embd.to_csv(out_path,index=False,index_label=False)
print("Finished")
