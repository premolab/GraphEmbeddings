import pandas as pd

from codes.Experiment import get_easy_datasets
from codes.log_progress import log_progress
from codes.models.HistLossGraph2vec import HistLossGraph2vec
from codes.utils import vis_embeddings, make_gif


def draw(nx_G, Y, name):
    folder = './emb/interns/{}/'.format(name)
    filepath_temp = folder+'hist_'+name+'{}.tmp_emb'
    for i in log_progress(#list(range(1, 20, 1)) +
                          list(range(1021, 10000, 10)),
                          #list(range(101, 20002, 50)),
                          every=1, add_val=True):
        try:
            embeding_filename = filepath_temp.format(str(i).zfill(6))
            nodes, embedding = HistLossGraph2vec.load(embeding_filename)
            vis_embeddings(nx_G, pd.DataFrame(embedding.T, columns=nodes), name, Y[0], i)
        except:
            continue
    make_gif('../histplot/{}/'.format(name))

if __name__ == '__main__':
    for dataset in get_easy_datasets():
        dataset = list(dataset)
        dataset[2] = dataset[2] + '_new_hist'
        print(dataset[2])
        draw(*dataset)
