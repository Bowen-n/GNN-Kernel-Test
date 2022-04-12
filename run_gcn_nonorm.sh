# gcn_nonorm is w + aggr (add, mean, max)
cd core
python train.py -d PubMed -m gcn_nonorm -l 3 -c 3 --aggr add --bn &&\
python train.py -d PubMed -m gcn_nonorm -l 3 -c 3 --aggr mean --bn &&\
python train.py -d PubMed -m gcn_nonorm -l 3 -c 3 --aggr max --bn &&\

python train.py -d PubMed -m gcn_nonorm -l 3 -c 3 --aggr add &&\
python train.py -d PubMed -m gcn_nonorm -l 3 -c 3 --aggr mean &&\
python train.py -d PubMed -m gcn_nonorm -l 3 -c 3 --aggr max &&\

python train.py -d Cora -m gcn_nonorm -l 3 -c 3 --aggr add --bn &&\
python train.py -d Cora -m gcn_nonorm -l 3 -c 3 --aggr mean --bn &&\
python train.py -d Cora -m gcn_nonorm -l 3 -c 3 --aggr max --bn &&\

python train.py -d Cora -m gcn_nonorm -l 3 -c 3 --aggr add &&\
python train.py -d Cora -m gcn_nonorm -l 3 -c 3 --aggr mean &&\
python train.py -d Cora -m gcn_nonorm -l 3 -c 3 --aggr max &&\

python train.py -d CiteSeer -m gcn_nonorm -l 3 -c 3 --aggr add --bn &&\
python train.py -d CiteSeer -m gcn_nonorm -l 3 -c 3 --aggr mean --bn &&\
python train.py -d CiteSeer -m gcn_nonorm -l 3 -c 3 --aggr max --bn &&\

python train.py -d CiteSeer -m gcn_nonorm -l 3 -c 3 --aggr add &&\
python train.py -d CiteSeer -m gcn_nonorm -l 3 -c 3 --aggr mean &&\
python train.py -d CiteSeer -m gcn_nonorm -l 3 -c 3 --aggr max
