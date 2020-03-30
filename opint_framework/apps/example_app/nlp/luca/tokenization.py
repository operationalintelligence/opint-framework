from opint_framework.core.nlp.nlp import Tokenization
from text_parsing_utils import split_urls, clean_tokens


class LucaTokenization(Tokenization):

    def __init__(self, ctx, err_col="t__error_message", id_col="msg_id"):
        super(LucaTokenization, self).__init__(ctx)
        self.ctx.err_col = err_col
        self.ctx.id_col = id_col

    def tokenize_messages(self, **kwargs):
        err_col = self.ctx['err_col']
        id_col = self.ctx['id_col']
        dataset = self.ctx['dataset']

        import pyspark.sql.functions as F
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType, ArrayType
        from pyspark.ml.feature import Tokenizer, StopWordsRemover
        from pyspark.ml import Pipeline

        # transform in user defined function
        split_urls_udf = udf(split_urls, StringType())

        # split urls appropriately
        test_data = dataset.select(id_col, err_col).withColumn("corrected_message", split_urls_udf(err_col))

        # split text into tokens
        tokenizer = Tokenizer(inputCol="corrected_message", outputCol="tokens")
        vector_data = tokenizer.transform(test_data)

        # transform in user defined function
        clean_tokens_udf = udf(lambda entry: clean_tokens(entry, custom_split=True), ArrayType(StringType()))

        # clean tokens
        vector_data = vector_data.withColumn("tokens_cleaned", clean_tokens_udf("tokens"))

        # remove stop (common, non-relevant) words
        stop_remove = StopWordsRemover(inputCol="tokens_cleaned", outputCol="stop_token")
        stop_remove1 = StopWordsRemover(inputCol="stop_token", outputCol="stop_token_1", stopWords=["", ":", "-", "+"])

        data_prep_pipeline = Pipeline(stages=[stop_remove, stop_remove1])

        pipeline_executor = data_prep_pipeline.fit(vector_data)
        vector_data = pipeline_executor.transform(vector_data)

        return (vector_data)

    def detokenize_messages(self, tokenizer, tokenized):
        pass

    def clean_tokens(self, tokenized):
        pass

    def get_vocabulary(self, tokenized):
        pass

    def tokenize_string(self, tokenizer, row):
        pass

    def detokenize_string(self, tokenizer, sequence):
        pass