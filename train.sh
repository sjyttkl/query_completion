# ./ trainer.py  path / to / expdir - -data / path / to / data.tsv - -valdata / path / to / valdata.tsv

source ~/.bash_profile

dir=/Users/songdongdong/PycharmProjects/query_completion/model
data=/Users/songdongdong/workSpace/datas/aol_search_query_logs/user-ct-test-collection-02.txt
python code/trainer.py dir --data data