from opint_framework.apps.example_app.nlp.pyspark_based.pyspark_nlp_adapter import pysparkNLPAdapter
from opint_framework.core.dataproviders.hdfs_provider import HDFSLoader

data_path = """/home/luca/PycharmProjects/opint-framework/opint_framework/apps/example_app/nlp/pyspark_based/sample_data/dump_1dec2020/*.json"""

# instantiate NLPAdapter object
pipeline = pysparkNLPAdapter(path_list=[data_path], vo=None, filter_T3=True,  # data
                             tks_col="stop_token_1",  # tokenization
                             w2v_model_path="results/w2v", w2v_mode="train", w2v_save_mode="overwrite",
                             emb_size=3, win_size=8, min_count=1, tks_vec="message_vector",  # word2vec
                             ft_col="features", kmeans_model_path="results/kmeans", kmeans_mode="train",
                             pred_mode="static", new_cluster_thresh=None, k_list=[2, 4],
                             distance="cosine", opt_initSteps=10, opt_tol=0.01, opt_maxIter=10,
                             log_path=None, n_cores=4,  # K_optim
                             tr_initSteps=30, tr_tol=0.001, tr_maxIter=30,  # train_kmeans
                             clust_col="prediction", wrdcld=True, timeplot=True)

pipeline.context['dataset'] = HDFSLoader().pull_hdfs_json(pipeline.context['path_list'],
                                                          pipeline.context['spark'])

# retrieve just data
all_transfers = pipeline.context['dataset'].select("data.*")

# filter test_errors only
test_errors = all_transfers.filter(all_transfers["t_final_transfer_state_flag"] == 0)
if pipeline.context['vo'] is not None:
    test_errors = test_errors.filter(test_errors["vo"] == pipeline.context['vo'])

# add row id and select only relevant variables
test_errors = test_errors.select(f"{pipeline.context['id_col']}", "t__error_message", "src_hostname",
                                 "dst_hostname", f"{pipeline.context['timestamp_tr_x']}")

from opint_framework.apps.example_app.nlp.pyspark_based.utils import convert_timestamp_to_datetime

test_errors = convert_endpoint_to_site(test_errors)
