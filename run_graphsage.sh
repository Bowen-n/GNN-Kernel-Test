# graphsage is aggr+w (add, mean, max)
cd core
python train.py -d PubMed -m graphsage -l 3 -c 3 --aggr add --bn &&\
python train.py -d PubMed -m graphsage -l 3 -c 3 --aggr mean --bn &&\
python train.py -d PubMed -m graphsage -l 3 -c 3 --aggr max --bn &&\

python train.py -d PubMed -m graphsage -l 3 -c 3 --aggr add &&\
python train.py -d PubMed -m graphsage -l 3 -c 3 --aggr mean &&\
python train.py -d PubMed -m graphsage -l 3 -c 3 --aggr max &&\

python train.py -d Cora -m graphsage -l 3 -c 3 --aggr add --bn &&\
python train.py -d Cora -m graphsage -l 3 -c 3 --aggr mean --bn &&\
python train.py -d Cora -m graphsage -l 3 -c 3 --aggr max --bn &&\

python train.py -d Cora -m graphsage -l 3 -c 3 --aggr add &&\
python train.py -d Cora -m graphsage -l 3 -c 3 --aggr mean &&\
python train.py -d Cora -m graphsage -l 3 -c 3 --aggr max &&\

python train.py -d CiteSeer -m graphsage -l 3 -c 3 --aggr add --bn &&\
python train.py -d CiteSeer -m graphsage -l 3 -c 3 --aggr mean --bn &&\
python train.py -d CiteSeer -m graphsage -l 3 -c 3 --aggr max --bn &&\

python train.py -d CiteSeer -m graphsage -l 3 -c 3 --aggr add &&\
python train.py -d CiteSeer -m graphsage -l 3 -c 3 --aggr mean &&\
python train.py -d CiteSeer -m graphsage -l 3 -c 3 --aggr max
