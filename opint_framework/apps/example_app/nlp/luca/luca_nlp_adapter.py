from pyspark.shell import spark
from pyspark.sql import SparkSession
from opint_framework.apps.example_app.nlp.luca.tokenization import LucaTokenization
from opint_framework.apps.example_app.nlp.luca.vectorization import LucaVectorization
from opint_framework.apps.example_app.nlp.luca.clustering import LucaClustering
from opint_framework.core.nlp.nlp import NLPAdapter
import pyspark.sql.functions as F


class LucaNLPAdapter(NLPAdapter):

    def __init__(self, path_list, vo, id_col="msg_id", timestamp_tr_x="timestamp_tr_comp", tks_col="stop_token_1",
                 w2v_model_path=None, ft_col="features", kmeans_model_path=None, kmeans_mode="train",
                 pred_mode="static", new_cluster_thresh=None, k_list=[12, 14, 16, 18, 20],  # update_model_path=None,
                 distance="cosine", opt_initSteps=10, opt_tol=0.0001, opt_maxIter=30, log_path=None, n_cores=5, # K_optim
                 tr_initSteps=200, tr_tol=0.000001, tr_maxIter=100,  # train_kmeans
                 )
        ):
        NLPAdapter.__init__(self,
                            tokenization=LucaTokenization(self.context),
                            vectorization=LucaVectorization(self.context),
                            clusterization=LucaClustering(self.context))
        self.context['spark'] = SparkSession.builder.master("local[*]").appName("sample_app_inference").getOrCreate()
        self.context['path_list'] = path_list
        self.context['id_col'] = id_col
        self.context['timestamp_tr_x'] = timestamp_tr_x
        self.context['vo'] = vo
        self.context['dataset'] = None
        self.context['tks_col'] = tks_col
        self.context['ft_col'] = ft_col
        self.context['w2v_model_path'] = w2v_model_path
        self.context['kmeans_model_path'] = kmeans_model_path
        self.context['pred_mode'] = pred_mode
        self.context['kmeans_mode'] = kmeans_mode
        self.context['new_cluster_threshold'] = new_cluster_threshold
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


def pre_process(self):
        self.context['all_transfers'] = spark.read.json(self.context['path_list']) # WHY WE INTRODUCE NEW KEY?

        # retrieve just data
        all_transfers = self.context['all_transfers'].select("data.*")

        # filter test_errors only
        test_errors = all_transfers.filter(all_transfers["t_final_transfer_state_flag"] == 0)
        if self.context['vo'] is not None:
            test_errors = test_errors.filter(test_errors["vo"] == self.context['vo'])

        # add row id and select only relevant variables
        test_errors = test_errors.withColumn(f"{self.context['id_col']}", F.monotonically_increasing_id()).select(
            f"{self.context['id_col']}", "t__error_message", "src_hostname", "dst_hostname",
            f"{self.context['timestamp_tr_x']}")

        self.context['dataset'] = test_errors

    def run(self):
        # LUCA: consider the following:
        # 1) add base_w2v_path and data_window as arguments;
        # 2) add w2v hyperparams as arguments;
        # 3) add w2v_mode argument ("train" or "load") for training/inference modes
        # 4) add kmeans_mode argument ("train" or "load", currently just "train" available)

        token_data = self.tokenization.tokenize_messages()

        # setup w2v model paths
        base_w2v_path = "results/sample_app/" # NOTE: this is a Hadoop path
        data_window = "9-13mar2020"
        emb_size = 150
        min_count = 500
        win_size = 8
        w2v_path = "{}/w2v_models/data_window_{}/w2v_VS={}_MC={}_WS={}".format(base_w2v_path, data_window, emb_size, min_count, win_size)

        if False: # substitute with: w2v_mode=="train":
            vector_data = self.vectorization.train_model(token_data, path_to_model=w2v_path, tks_col="stop_token_1",
                                                         id_col=id_col, out_col='message_vector', embedding_size=emb_size,
                                                         window=win_size, min_count=min_count, workers=12, mode="new")
        elif True: # substitute with: w2v_mode=="load":
            w2v_model = self.vectorization.load_model(w2v_path)
            vector_data = w2v_model.transform(tks_data)

        vector_data = self.clusterization.data_preparation(vector_data, self.context['tks_col'])

        # train for different Ks
        res = self.clusterization.K_optim(self.context['k_list'], dataset=vector_data, tks_vec=self.context['tks_col'],
                                          ft_col=self.context['ft_col'], distance=self.context['distance'],
                                          initSteps=self.context['opt_initSteps'], tol=self.context['opt_tol'],
                                          maxIter=self.context['opt_maxIter'], n_cores=self.context['n_cores'], log_path=log_path)

        k_sil = get_k_best(res, "silhouette")

        if pred_mode == "update":
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

        best_k_log_path = Path(log_path).parent / "best_K={}.txt".format(k_sil)
        original_data = kmeans_preproc(original_data, tks_vec)
        kmeans_model = train_kmeans(original_data, ft_col=ft_col, k=k_sil, distance=distance,
                                    initSteps=tr_initSteps, tol=tr_tol, maxIter=tr_maxIter,
                                    save_path=kmeans_model_path, mode=save_mode, log_path=best_k_log_path)

    original_data = kmeans_predict(original_data, kmeans_model["model"], pred_mode=pred_mode,
                                   new_cluster_thresh=None, update_model_path=kmeans_model_path)
    def post_process(self):
        pass
