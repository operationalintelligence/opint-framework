

def w2v_preproc(original_data, msg_col, id_col, model_path):
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
    original_data = tokenizer(original_data, err_col=msg_col, id_col=id_col)
    w2v_model = load_w2v(model_path)
    original_data = w2v_model.transform(original_data)
    return (original_data)