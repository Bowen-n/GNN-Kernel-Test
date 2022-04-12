cd core
python train.py -d PubMed -m gcn -l 3 -c 3 --aggr add --bn &&\
python train.py -d PubMed -m gcn -l 3 -c 2 --aggr add --bn &&\
python train.py -d PubMed -m gcn -l 3 -c 1 --aggr add --bn &&\

python train.py -d PubMed -m gcn -l 3 -c 3 --aggr add &&\
python train.py -d PubMed -m gcn -l 3 -c 2 --aggr add &&\
python train.py -d PubMed -m gcn -l 3 -c 1 --aggr add &&\

python train.py -d Cora -m gcn -l 3 -c 3 --aggr add --bn &&\
python train.py -d Cora -m gcn -l 3 -c 2 --aggr add --bn &&\
python train.py -d Cora -m gcn -l 3 -c 1 --aggr add --bn &&\

python train.py -d Cora -m gcn -l 3 -c 3 --aggr add &&\
python train.py -d Cora -m gcn -l 3 -c 2 --aggr add &&\
python train.py -d Cora -m gcn -l 3 -c 1 --aggr add &&\

python train.py -d CiteSeer -m gcn -l 3 -c 3 --aggr add --bn &&\
python train.py -d CiteSeer -m gcn -l 3 -c 2 --aggr add --bn &&\
python train.py -d CiteSeer -m gcn -l 3 -c 1 --aggr add --bn &&\

python train.py -d CiteSeer -m gcn -l 3 -c 3 --aggr add &&\
python train.py -d CiteSeer -m gcn -l 3 -c 2 --aggr add &&\
python train.py -d CiteSeer -m gcn -l 3 -c 1 --aggr add
