from opint_framework.core.nlp.nlp import Vectorization


class LucaVectorization(Vectorization):

    def __init__(self, ctx, tokenization):
        super(LucaVectorization, self).__init__(ctx)
        self.tokenization = tokenization

    def train_model(self, messages, path_to_model=None, embedding_size=150, window=8, min_count=500, workers=12,
                    # training_algorithm=None, iter=None, # unused parent params
                    # additional params
                    tks_col="stop_token_1", id_col="msg_id", out_col='message_vector', mode="new"
                    ):
        """Train Word2Vec model on the input tokens column.

        -- params:
        messages (pyspark.sql.dataframe.DataFrame): data frame with at least error string and id columns
        tks_col (string): name of the column containing the lists of tokens to feed into the word2vec model
        id_col (string): name of the message id column
        out_col (string): name of the output column for the word2vec vector representation of the messages
        embedding_size (int): dimension of the word2vec embedded space
        min_count (int): minimum frequency for tokens to be considered in the training
        window (int): window size for word2vec model
        path_to_model (string): path where to save the trained model. Default is None (no saving)
        mode ("new" or "overwrite"): whether to save new file or overwrite pre-existing one.
        workers (int): number of cores for word2vec training

        Returns:
        model (pyspark.ml.feature.Word2VecModel): trained Word2vec model
        """
        from pyspark.ml.feature import Word2Vec
        from pathlib import Path

        # base_w2v_path = "results/sample_app/" # NOTE: this is a Hadoop path
        # data_window = "9-13mar2020"
        # emb_size = self.ctx['emb_size']
        # min_count = self.ctx['min_count']
        # win_size = self.ctx['win_size']
        # path_to_model = "{}/w2v_VS={}_MC={}_WS={}".format(self.ctx['w2v_path'], emb_size, min_count, win_size)

        # intialise word2vec
        word2vec = Word2Vec(vectorSize=embedding_size, minCount=min_count, windowSize=window,
                            inputCol=tks_col, outputCol=out_col, numPartitions=workers)

        train_data = messages.select(id_col, tks_col)
        model = word2vec.fit(train_data)

        if path_to_model:
            p = Path(path_to_model)
            p.parent.mkdir(parents=True, exist_ok=True)
            outname = "{}/w2v_sample_app_example_VS={}_MC={}_WS={}".format(path_to_model, embedding_size, min_count, window)
            self.ctx["w2v_model_path"] = outname
            print("Saving w2v model to: {}".format(outname))
            if mode == "overwrite":
                model.write().overwrite().save(outname)
            else:
                model.save(outname)

        return (model)

    def load_model(self, path_to_model):
        """Load Word2Vec model from model_path."""
        from pyspark.ml.feature import Word2VecModel

        w2vec_model = Word2VecModel.load(path_to_model)
        return(w2vec_model)

    def update_model(self, path_to_model, tokenized):
        pass

    def vectorize_messages(self, word2vec, tokenized): # LUCA: I'd probably change into w2v_path, tokenized, so the function
                                                        # loads pre-trained w2v and gets vector representation for tokenized df
        """ Return word2vec emdeding of input tokens."""
        vector_data = word2vec.transform(tokenized)

        return(vector_data)

    def w2v_preproc(self, original_data, msg_col, id_col, model_path):
        """Take input dataset as extracted from hdfs and compute Word2Vec representation.

        -- params:
        original_data (pyspark.sql.dataframe.DataFrame): data frame with at least error string and id columns
        msg_col (string): name of the error string column
        id_col (string): name of the message id column
        model_path (string): path where to load pre-trained word2vec model

        Returns:
        original_data (pyspark.ml.feature.Word2VecModel): the original data with an extra
                        "message_vector" column with word2vec embedding
        """
        original_data = self.tokenization.tokenize_messages(original_data, err_col=msg_col, id_col=id_col)
        w2v_model = self.load_w2v(model_path)
        original_data = w2v_model.transform(original_data)
        return (original_data)