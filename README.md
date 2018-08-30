# GraphEmbeddings
This repository contains realization of `DDoS` (aka `Histogram loss`)
algorithm for generating graph embeddings.

## Structure
All code is stored in folder `final_src`

Realization of several embedding algorithm including our algorithm
can be found in folder `transformers`.

Folder `io_utils` contains code responsible for reading
and writing graphs and embeddings.

Folder `transformation` contains generic code to generate an embedding
with any available algorithm.

Other folders represent sets of experiments for comparing algorithms:
`link_prediction`, `classification` and `clusterization`.

## How to run
To make this code work you need to replace path to necessary dataset
in file `final_src/settings.py` with your local path.