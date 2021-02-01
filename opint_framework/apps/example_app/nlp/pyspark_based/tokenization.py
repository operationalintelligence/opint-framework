from opint_framework.core.nlp.nlp import Tokenization
from opint_framework.apps.example_app.nlp.pyspark_based.text_parsing_utils import split_urls, clean_tokens


class pysparkTokenization(Tokenization):

    def __init__(self, ctx):
        super(pysparkTokenization, self).__init__(ctx)

    def tokenize_messages(self, **kwargs):
        """Take input message and split it into tokens.

            -- params (given by the parent class pysparkTokenization):
            dataset (pyspark.sql.dataframe.DataFrame): data frame with at least error string and id columns
            err_col (string): name of the error string column
            id_col (string): name of the message id column

            Returns:
            vector_data (pyspark.sql.dataframe.DataFrame): data frame with id_col, err_col and additional tokenization steps:
                                             corrected_message --> string with corrected urls
                                             tokens --> list of tokens taken from corrected_message
                                             tokens_cleaned --> list of tokens cleaned from punctuation and empty entries
                                             stop_token --> list of tokens after removing common english stopwords
                                             stop_token_1 --> list of tokens after removing custom stopwords, i.e. ["", ":", "-", "+"]
        """
        err_col = self.ctx['err_col']
        id_col = self.ctx['id_col']
        tks_col = self.ctx['tks_col']
        dataset = self.ctx['dataset']

        import pyspark.sql.functions as F
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType, ArrayType
        from pyspark.ml.feature import Tokenizer, StopWordsRemover
        from pyspark.ml import Pipeline

        # transform in user defined function
        split_urls_udf = udf(split_urls, StringType())

        # split urls appropriately
        test_data = dataset.withColumn("corrected_message", split_urls_udf(err_col))

        # split text into tokens
        tokenizer = Tokenizer(inputCol="corrected_message", outputCol="tokens")
        token_data = tokenizer.transform(test_data)

        # transform in user defined function
        clean_tokens_udf = udf(lambda entry: clean_tokens(entry, custom_split=True), ArrayType(StringType()))

        # clean tokens
        token_data = token_data.withColumn("tokens_cleaned", clean_tokens_udf("tokens"))

        # remove stop (common, non-relevant) words
        stop_remove = StopWordsRemover(inputCol="tokens_cleaned", outputCol="stop_token")
        stop_remove1 = StopWordsRemover(inputCol="stop_token", outputCol=tks_col, stopWords=["", ":", "-", "+"])

        data_prep_pipeline = Pipeline(stages=[stop_remove, stop_remove1])

        pipeline_executor = data_prep_pipeline.fit(token_data)
        token_data = pipeline_executor.transform(token_data)

        return (token_data)

    def detokenize_messages(self, tokenized, tks_col, out_detoken_col):
        """Takes pyspark dataframe \"tokenized\" where \"err_col\" contains list of tokens
        and return a dataframe with the additional \"message_string\" column where tokens are joint back.
        """
        # import pyspark.sql.functions as F
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        # transform row entry from list of tokens to string
        detokenize_udf = udf(lambda entry: " ".join(entry), StringType())

        # detokenize
        tokenized = tokenized.withColumn(out_detoken_col, detokenize_udf(tks_col))

        return(tokenized)

    def clean_tokens(self, tokenized):
        pass

    def get_vocabulary(self, tokenized):
        raise NotImplementedError

    def tokenize_string(self, tokenizer, row):
        pass

    def detokenize_string(self, tokenizer, sequence):
        pass

    def abstract_params(self, dataset, tks_col="tokens_cleaned", out_col="abstract_message"):
        """Abstract parameters from a column of tokens lists.

        -- params:
        dataset (pyspark.sql.dataframe.DataFrame): data frame with at least a column containg lists of tokens
        tks_col (string): name of the tokens lists column
        out_col (string): name of the column where to store abstracted tokens

        Returns:
        dataset (pyspark.sql.dataframe.DataFrame): the input dataset with an extra
                        out_col column with abstracted tokens
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType, ArrayType
        from opint_framework.apps.example_app.nlp.pyspark_based.abstraction_utils import abstract_message
        # transform in user defined function
        abstract_message_udf = udf(abstract_message, ArrayType(StringType()))

        dataset = dataset.withColumn(out_col, abstract_message_udf(tks_col))
        return (dataset)