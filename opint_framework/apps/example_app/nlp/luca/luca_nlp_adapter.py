from pyspark.shell import spark
from pyspark.sql import SparkSession
from opint_framework.apps.example_app.nlp.luca.tokenization import LucaTokenization
from opint_framework.apps.example_app.nlp.luca.vectorization import LucaVectorization
from opint_framework.apps.example_app.nlp.luca.clustering import LucaClustering
from opint_framework.core.nlp.nlp import NLPAdapter
import pyspark.sql.functions as F


class LucaNLPAdapter(NLPAdapter):

    def __init__(self, path_list, vo, id_col="msg_id", timestamp_tr_x="timestamp_tr_comp"):
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

    def pre_process(self):
        self.context['all_transfers'] = spark.read.json(self.context['path_list'])

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

        vector_data = self.clusterization.data_preparation()

    def post_process(self):
        pass
