from node_classification.main import main as clasMain
from node_clusterization.main import main as clusMain
from link_prediction.main import main as linkMain


def main():
    clasMain()
    clusMain()
    linkMain()


if __name__ == '__main__':
    main()
