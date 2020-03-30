from pyspark.shell import spark
from pyspark.sql import SparkSession
from opint_framework.apps.example_app.nlp.luca.tokenization import LucaTokenization
from opint_framework.core.nlp.nlp import NLPAdapter
import pyspark.sql.functions as F


class LucaNLPAdapter(NLPAdapter):

    def __init__(self, path_list, vo, id_col="msg_id", timestamp_tr_x="timestamp_tr_comp"):
        NLPAdapter.__init__(self,
                            tokenization=LucaTokenization(self.context),
                            vectorization=None,
                            clusterization=None)
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
        vector_data = self.tokenization.tokenize_messages()


    def post_process(self):
        pass
