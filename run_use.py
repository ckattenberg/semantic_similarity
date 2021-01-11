from embed import universal_sentence_encoder as use
from process import readdata
import pandas as pd
import numpy as np

def embed_data(data):
    q1 = data["question1"]
    q2 = data["question2"]
    em1 = q1.apply(use.encode)
    em2 = q2.apply(use.encode)
    em_df = pd.DataFrame({"em1": em1, "em2": em2})
    return em_df

def sim_data(em_df):
    similarities = em_df.apply(lambda row:use.cossim(row.em1, row.em2), axis=1)
    return similarities

def compare(sim, is_duplicate):
    threshold = 0.9
    tp = sum(np.where((sim >= threshold) & (is_duplicate == 1), 1, 0))
    tf = sum(np.where((sim < threshold) & (is_duplicate == 0), 1, 0))
    fp = sum(np.where((sim >= threshold) & (is_duplicate == 0), 1, 0))
    fn = sum(np.where((sim < threshold) & (is_duplicate == 1), 1, 0))

    print("Accuracy test: ", (tp+tf)/len(sim))
    print("Precision test: ", tp/(tp+fp))
    print("Recall test: ", tp/(tp + fn))
    return tp, tf, fp, fn


if __name__ == "__main__":
    data = readdata.read()
    em_df = embed_data(data)
    sims = sim_data(em_df)
    dups = data["is_duplicate"]
    compare(sims, dups)