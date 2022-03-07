echo "Set FLASK_ENV"
export FLASK_ENV"=development"
echo "Set FLASK_APP"
export FLASK_APP="week3"

#########
# command line commands - saved here for convenience, not for running script
#######

# level 1 - content classification
python week3/createContentTrainingData.py --output /workspace/datasets/categories/output.fasttext --min_products 50
alias prep_data="python week3/createContentTrainingData.py --output /workspace/datasets/categories/output.fasttext"

shuf /workspace/datasets/categories/output.fasttext --output /workspace/datasets/categories/shuffled-output.fasttext 

cd /workspace/datasets/categories/
head -n -10000 shuffled-output.fasttext > data.train
tail -n -10000 shuffled-output.fasttext > data.test

# train
cd /workspace/search_with_machine_learning_course/week3

# run 1 
alias train_model="~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/categories/data.train -output pray_for_ukraine --min_products 50"
"-lr 1.0 -epoch 25 -wordNgrams 2"

# predict
alias predict_output=~/fastText-0.9.2/fasttext predict pray_for_ukraine.bin -

# test
alias test_model="~/fastText-0.9.2/fasttext test pray_for_ukraine.bin /workspace/datasets/categories/data.test"
alias test_model_top5="~/fastText-0.9.2/fasttext test pray_for_ukraine.bin /workspace/datasets/categories/data.test 5"
alias test_model_top10="~/fastText-0.9.2/fasttext test pray_for_ukraine.bin /workspace/datasets/categories/data.test 10"

~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/titles.txt -output /workspace/datasets/fasttext/title_model -minCount 50 -epoch 10 -wordNgrams 2


~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/titles.txt -output /workspace/datasets/fasttext/title_model


# level 2 - synonyms
python extractTitles.py 

#
python extractTitles.py --sample_rate 0.5

# unsupervised model, skip_gram is preferred over cbow by many b/c it uses contextual info 
alias skip_gram="~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/titles.txt -output /workspace/datasets/fasttext/title_model"

alias run_title_model="~/fastText-0.9.2/fasttext nn /workspace/datasets/fasttext/title_model.bin"


$ ~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/titles.txt -output /workspace/datasets/fasttext/title_model -epoch 30 -minCount 50 -loss hs