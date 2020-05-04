from pyspark.shell import spark
from pyspark.sql import SparkSession
from opint_framework.apps.example_app.nlp.pyspark_based.tokenization import pysparkTokenization
from opint_framework.apps.example_app.nlp.pyspark_based.vectorization import pyspark_w2v_Vectorization
from opint_framework.apps.example_app.nlp.pyspark_based.clustering import pyspark_KM_Clustering
from opint_framework.core.nlp.nlp import NLPAdapter
from opint_framework.core.dataproviders.hdfs_provider import HDFSLoader
import pyspark.sql.functions as F


class pysparkNLPAdapter(NLPAdapter):

    def __init__(self, path_list, vo, id_col="msg_id", timestamp_tr_x="timestamp_tr_comp",  ## original data
                 tks_col="stop_token_1",  ## tokenization
                 w2v_model_path=None, w2v_mode="load", w2v_save_mode="new", emb_size=150, win_size=8, min_count=500,
                 tks_vec="message_vector",  ## word2vec
                 ft_col="features", kmeans_model_path=None, kmeans_mode="train",
                 pred_mode="static", new_cluster_thresh=None, k_list=[12, 14, 16, 18, 20],  # update_model_path=None,
                 distance="cosine", opt_initSteps=10, opt_tol=0.0001, opt_maxIter=30, log_path=None, n_cores=5,
                 ## K_optim
                 tr_initSteps=200, tr_tol=0.000001, tr_maxIter=100,  ## train_kmeans
                 clust_col="prediction"  # visualization)
                 ):
        # self.context = {}

        super(pysparkNLPAdapter, self).__init__(name="PySpark_adapter"  # ,
                                                # tokenization=self.tokenization,
                                                # vectorization=pyspark_w2v_Vectorization(self.context, self.tokenization),
                                                # clusterization=pyspark_KM_Clustering(self.context))
                                                )
        self.tokenization = pysparkTokenization(self.context)
        self.vectorization = pyspark_w2v_Vectorization(self.context, self.tokenization)
        self.clusterization = pyspark_KM_Clustering(self.context)
        self.context['spark'] = SparkSession.builder.master("local[*]").appName("sample_app_inference").getOrCreate()
        self.context['path_list'] = path_list
        self.context['err_col'] = "t__error_message"
        self.context['id_col'] = id_col
        self.context['timestamp_tr_x'] = timestamp_tr_x
        self.context['vo'] = vo
        self.context['dataset'] = None
        self.context['tks_col'] = tks_col
        self.context['tks_vec'] = tks_vec
        self.context['ft_col'] = ft_col
        self.context['w2v_model_path'] = w2v_model_path
        self.context['w2v_mode'] = w2v_mode
        self.context['w2v_save_mode'] = w2v_save_mode
        self.context['emb_size'] = emb_size
        self.context['win_size'] = win_size
        self.context['min_count'] = min_count
        self.context['kmeans_model_path'] = kmeans_model_path
        self.context['pred_mode'] = pred_mode
        self.context['kmeans_mode'] = kmeans_mode
        self.context['new_cluster_threshold'] = new_cluster_thresh
        self.context['k_list'] = k_list
        self.context['distance'] = distance
        self.context['opt_initSteps'] = opt_initSteps
        self.context['opt_tol'] = opt_tol
        self.context['opt_maxIter'] = opt_maxIter
        self.context['tr_initSteps'] = tr_initSteps
        self.context['tr_tol'] = tr_tol
        self.context['tr_maxIter'] = tr_maxIter
        self.context['log_path'] = log_path
        self.context['n_cores'] = n_cores
        self.context['clust_col'] = clust_col

    def pre_process(self):
        self.context['dataset'] = HDFSLoader().pull_hdfs_json(self.context['path_list'], self.context['spark'])
        # self.context['dataset'] = spark.read.json(
        #     self.context['path_list'])  # WHY WE INTRODUCE NEW KEY? -> changed all_tranfers with dataset

        # retrieve just data
        all_transfers = self.context['dataset'].select("data.*")

        # filter test_errors only
        test_errors = all_transfers  # .filter(all_transfers["t_final_transfer_state_flag"] == 0)
        if self.context['vo'] is not None:
            test_errors = test_errors  # .filter(test_errors["vo"] == self.context['vo'])

        # add row id and select only relevant variables
        test_errors = test_errors.select(  #.withColumn(f"{self.context['id_col']}", F.monotonically_increasing_id()).select(
            f"{self.context['id_col']}", "t__error_message", "src_hostname", "dst_hostname",
            f"{self.context['timestamp_tr_x']}")

        self.context['dataset'] = test_errors

    def run(self):
        from opint_framework.apps.example_app.nlp.pyspark_based.kmeans import get_k_best
        from pathlib import Path

        token_data = self.tokenization.tokenize_messages()

        if self.context['w2v_mode'] == "train":
            w2v_model = self.vectorization.train_model(token_data, path_to_model=self.context['w2v_model_path'],
                                                       tks_col=self.context['tks_col'],
                                                       id_col=self.context['id_col'], out_col=self.context['tks_vec'],
                                                       embedding_size=self.context['emb_size'],
                                                       window=self.context['win_size'],
                                                       min_count=self.context['min_count'],
                                                       workers=self.context['n_cores'],
                                                       mode=self.context["w2v_save_mode"])
            vector_data = w2v_model.transform(token_data)

        elif self.context['w2v_mode'] == "load":
            w2v_model = self.vectorization.load_model(self.context['w2v_model_path'])
            vector_data = w2v_model.transform(token_data)
        # return(vector_data)
        # vector_data = self.clusterization.data_preparation(vector_data, self.context['tks_vec'])

        # K value optimization
        res = self.clusterization.K_optim(k_list=self.context['k_list'], messages=vector_data,
                                          tks_vec=self.context['tks_vec'],
                                          ft_col=self.context['ft_col'], distance=self.context['distance'],
                                          initSteps=self.context['opt_initSteps'], tol=self.context['opt_tol'],
                                          maxIter=self.context['opt_maxIter'], n_cores=self.context['n_cores'],
                                          log_path="{}/K-Means_optimization.txt".format(self.context['log_path']))

        k_sil = get_k_best(res, "silhouette")

        # setup prediciton mode params
        if self.context['pred_mode'] == "update":
            save_mode = "overwrite"
            kmeans_model_path = "temp_ciccio"
        #         elif kmeans_mode=="load":
        #             kmeans_model_path = None
        #             save_mode = "new"
        #         else:
        #             save_mode = "new"
        else:
            kmeans_model_path = None
            save_mode = "new"

        if self.context['log_path']:
            best_k_log_path = Path(self.context['log_path']).parent / "best_K={}.txt".format(k_sil)
        else:
            best_k_log_path = None

        # transform data into clustering suitable format
        vector_data = self.clusterization.data_preparation(vector_data, self.context['tks_vec'])

        # train best K
        kmeans_model = self.clusterization.train_model(messages=vector_data, ft_col=self.context['ft_col'], k=k_sil,
                                                       distance=self.context['distance'],
                                                       initSteps=self.context['tr_initSteps'],
                                                       tol=self.context['tr_tol'], maxIter=self.context['tr_maxIter'],
                                                       path_to_model=kmeans_model_path, mode=save_mode,
                                                       log_path=best_k_log_path)

        return (kmeans_model)

    def post_process(self, model, test_predictions=None):
        from opint_framework.apps.example_app.nlp.pyspark_based.cluster_visualization import summary

        # initialize to None in case not retrievable from the model
        best_k = None

        if type(model) == type({}):
            model = model["model"]
            test_predictions = model.summary.predictions
            best_k = model.summary.k
        elif not test_predictions:
            # first we have to pre-preocess hdfs data to get clustering suitable format, i.e.:
            # 1. Tokenize
            # 2. Vectorize
            # 3. Clustering.data_preparation
            test_predictions = self.tokenization.tokenize_messages()
            w2v_model = self.vectorization.load_model(self.context['w2v_model_path'])
            test_predictions = w2v_model.transform(test_predictions)
            test_predictions = self.clusterization.data_preparation(test_predictions, self.context['tks_vec'])
            ## WARNING: IF WE WANT TO USE ONLY POST PROCESSING WE NEED TO TAKE CARE OF INPUT PARAMETERS

            test_predictions = self.clusterization.predict(tokenized=test_predictions, model=model,
                                                           pred_mode=self.context['pred_mode'],
                                                           # new_cluster_thresh=None, update_model_path=kmeans_model_path--> need to be defined
                                                           )
        abs_dataset, summary = summary(dataset=test_predictions, k=best_k,
                                       clust_col=self.context['clust_col'], tks_col=self.context['tks_col'],
                                       abs_tks_in="tokens_cleaned", abs_tks_out="abstract_tokens",
                                       abstract=True, n_mess=None, wrdcld=False,
                                       original=self.context['dataset'], src_col="src_hostname", n_src=None,
                                       dst_col="dst_hostname", n_dst=None, timeplot=False,
                                       time_col=self.context['timestamp_tr_x'],
                                       save_path="results/sample_app/K={}".format(best_k),
                                       tokenization=self.tokenization)
        return (summary)

    def execute(self):
        try:
            print(f"\nNLP Adapter - {self.name}: Pre Processing input data")
            self.pre_process()
        except Exception as error:
            print(f"\nNLP Adapter - {self.name}: Pre Processing failed: {str(error)}")
            traceback.print_exc()
        try:
            print(f"\nNLP Adapter - {self.name}: Executing vectorization, tokenization and clusterization")
            model = self.run()
        except Exception as error:
            print(f"\nNLP Adapter - {self.name}: Log analysis failed: {str(error)}")
            traceback.print_exc()
        try:
            print(f"\nNLP Adapter - {self.name}: Post processing")
            summary = self.post_process(model=model)
            return (summary)
        except Exception as error:
            print(f"\nNLP Adapter - {self.name}: Post Processing failed: {str(error)}")
            traceback.print_exc()
