def compute_metrics(model_list, i, metric, distance):
    from pyspark.ml.evaluation import ClusteringEvaluator

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator(distanceMeasure=distance)

    if metric == "wsse":
        res = model_list[i].summary.trainingCost
    elif metric == "asw":
        res = evaluator.evaluate(model_list[i].summary.predictions)
    else:
        print("WARNING: wrong metric specified. Use either \"wsse\" or \"asw\".")
        return (None)
    return (res)



def plot_metrics(results):
    """Plot the trends of evaluation metrics from the output of K_optim."""
    import numpy as np
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(22, 6))

    k_list = [mod.summary.k for mod in results["model"]]
    best_K_wsse = np.argmin(results["wsse"])
    best_K_silhouette = np.argmax(results["silhouette"])

    _ = plt.subplot(1, 2, 1)
    _ = plt.plot(k_list, np.log(results["wsse"]), '-D', markevery=[best_K_wsse], markerfacecolor='red', markersize=12)
    _ = plt.xlabel("K")
    _ = plt.ylabel("log(WSSE)")
    _ = plt.xticks(k_list)
    _ = plt.title("Within Groups Sum of Squares")

    _ = plt.subplot(1, 2, 2)
    _ = plt.plot(k_list, results["silhouette"], '-D', markevery=[best_K_silhouette], markerfacecolor='red',
                 markersize=12)
    _ = plt.xlabel("K")
    _ = plt.ylabel("ASW")
    _ = plt.xticks(k_list)
    _ = plt.title("Average Silhouette Width")
    _ = plt.show()
    return (None)


def get_k_best(results, metric="silhouette", method="knee"):
    """Return the best K value according to the specified metric."""
    import numpy as np
    from kneed import KneeLocator
    from matplotlib import pyplot as plt

    if method == 'best':
        best_K_wsse = results["model"][np.argmin(results["wsse"])].summary.k
        best_K_silhouette = [np.argmax(results["silhouette"])].summary.k
    elif method == 'knee':
        k_list = [model.summary.k for model in results['model']]
        wsse_values = results['wsse']
        silhouette_values = results['silhouette']
        best_K_wsse = KneeLocator(k_list, wsse_values, curve='convex', direction='decreasing').knee
        if best_K_wsse is None:
            best_K_wsse = results["model"][np.argmin(results["wsse"])].summary.k
        best_K_silhouette = KneeLocator(k_list, silhouette_values, curve='concave', direction='increasing').knee
        if best_K_silhouette is None:
            best_K_silhouette = [np.argmax(results["silhouette"])].summary.k
    else:
        print("Error: wrong method parameter. Specify \"knee\" (default) or \"best\".")

    if metric == "silhouette":
        return (best_K_silhouette)
    elif metric == "wsse":
        return (best_K_wsse)
    else:
        print("Error: wrong metric parameter. Specify \"silhouette\" (default) or \"wsse\".")
        return (None)


def merge_predictions(original, predictions, orig_id="msg_id", pred_id="msg_id",
                      out_col_list=["tokens_cleaned", "abstract_tokens", "features", "prediction"]):
    """Merge custering output with original messages (from hdfs)
                    containing additional information (e.g. src/dst sites, timestamp, ...).

    -- params:
    original (pyspark.sql.dataframe.DataFrame): data frame with dta from hdfs
    predictions (pyspark.sql.dataframe.DataFrame): data frame with clustering results
    orig_id (string): name of the message id column on hdfs data frame
    perd_id (string): name of the message id column on prediction data frame
    out_col_list (list[string]): list of names of the columns of the prediction data frame to append in the output

    Returns:
    merge_df (pyspark.sql.dataframe.DataFrame): merged data frame
    """
    import numpy as np
    import pyspark.sql.functions as F

    # create list of output columns names
    ## remove duplicate names from right table
    out_col_list = [predictions[col_name] for col_name in out_col_list if col_name not in original.columns]
    ## extract columns from original dataframe with proper format
    output_columns = [original[col_names] for col_names in original.columns]
    ## put left and right columns together in the desired order
    output_columns.extend(out_col_list)

    # join original data with predicted cluster labels
    merge_df = original.join(predictions, original[orig_id] == predictions[pred_id],
                             how="outer").select(output_columns).orderBy(F.col(orig_id))
    return (merge_df)

def kmeans_inference(original_data, msg_col, id_col, w2v_model_path, tks_vec, ft_col, kmeans_mode, kmeans_model_path,
                     pred_mode="static", new_cluster_thresh=None, k_list=[12, 16, 20],  # update_model_path=None,
                     distance="cosine", opt_initSteps=10, opt_tol=0.0001, opt_maxIter=30, log_path=None, n_cores=5,
                     # K_optim
                     tr_initSteps=200, tr_tol=0.000001, tr_maxIter=100,  # train_kmeans
                     ):
    from language_models import w2v_preproc
    from pyspark.ml.clustering import KMeansModel
    import time
    import datetime
    from pyspark.ml.evaluation import ClusteringEvaluator
    """Perform inference on new error messages (Note: only K-Means can be re-trained/updated).

    -- params:
    original_data (pyspark.sql.dataframe.DataFrame): data frame with at least error string and id columns
    msg_col (string): name of the error string column
    id_col (string): name of the message id column
    model_path (string): path where to load pre-trained word2vec model
    tks_vec (string): name of the word2vec representations column
    ft_col (string): name of the features column
    kmeans_mode (\"load\" or \"train\"): kmeans mode: \"load\" uses pre-trained model, while \"train\" performs online training
    kmeans_model_path (string): path to pre-trained model (Specify None for re-training)
    pred_mode (\"static\" or \"update\"): prediction mode: \"static\" does not allow for creating new clusters
    new_cluster_thresh (float): distance threshold: if closest centroid is more distant than new_cluster_thresh 
                                then a new cluster is created for the new observation
    k_list (list): grid of K values to try
    distance (\"euclidean\" or \"cosine\"): distance measure for the kmeans algorithm
    opt_initStep (int): number of different random intializations for the kmeans algorithm in the optimization phase
    opt_tol (int): tolerance for kmeans algorithm convergence in the optimization phase
    opt_maxIter (int): maximum number of iterations for the kmeans algorithm in the optimization phase
    n_cores (int): number of cores to use
    log_path (string): where to save optimization stats. Default None (no saving)
    tr_initStep (int): number of different random intializations for the kmeans algorithm in the training phase
    tr_tol (int): tolerance for kmeans algorithm convergence in the training phase
    tr_maxIter (int): maximum number of iterations for the kmeans algorithm in the training phase

    Returns:
    original_data (pyspark.sql.dataframe.DataFrame): the input data frame with an extra \"prediction\" column
    """
    from pathlib import Path

    if kmeans_mode not in ["load", "train"]:
        print("""WARNING: invalid param \"kmeans_mode\". Specify either \"load\" to train load a pre-trained model 
              or \"train\" to train it online.""")
        return (None)

    original_data = w2v_preproc(original_data, msg_col, id_col, w2v_model_path)

    if kmeans_mode == "load":
        original_data = kmeans_preproc(original_data, tks_vec)
        kmeans_model = KMeansModel.load(kmeans_model_path)
    else:
        # K_optim()
        # initialize a grid of K (number of clusters) values
        #         k_list = [12, 16, 20]

        # train for different Ks
        res = K_optim(k_list, messages=original_data, tks_vec=tks_vec, ft_col=ft_col,
                      distance=distance, initSteps=opt_initSteps, tol=opt_tol, maxIter=opt_maxIter,
                      n_cores=n_cores, log_path=log_path)

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
                                    path_to_model=kmeans_model_path, mode=save_mode, log_path=best_k_log_path)

    original_data = kmeans_predict(original_data, kmeans_model["model"], pred_mode=pred_mode,
                                   new_cluster_thresh=None, update_model_path=kmeans_model_path)

    return (original_data)


def kmeans_inference(original_data, msg_col, id_col, w2v_model_path, tks_vec, ft_col, kmeans_mode, kmeans_model_path,
                     pred_mode="static", new_cluster_thresh=None, k_list=[12, 16, 20],  # update_model_path=None,
                     distance="cosine", opt_initSteps=10, opt_tol=0.0001, opt_maxIter=30, log_path=None, n_cores=5,
                     # K_optim
                     tr_initSteps=200, tr_tol=0.000001, tr_maxIter=100,  # train_kmeans
                     ):
    """Perform inference on new error messages (Note: only K-Means can be re-trained/updated).

    -- params:
    original_data (pyspark.sql.dataframe.DataFrame): data frame with at least error string and id columns
    msg_col (string): name of the error string column
    id_col (string): name of the message id column
    model_path (string): path where to load pre-trained word2vec model
    tks_vec (string): name of the word2vec representations column
    ft_col (string): name of the features column
    kmeans_mode (\"load\" or \"train\"): kmeans mode: \"load\" uses pre-trained model, while \"train\" performs online training
    kmeans_model_path (string): path to pre-trained model (Specify None for re-training)
    pred_mode (\"static\" or \"update\"): prediction mode: \"static\" does not allow for creating new clusters
    new_cluster_thresh (float): distance threshold: if closest centroid is more distant than new_cluster_thresh 
                                then a new cluster is created for the new observation
    k_list (list): grid of K values to try
    distance (\"euclidean\" or \"cosine\"): distance measure for the kmeans algorithm
    opt_initStep (int): number of different random intializations for the kmeans algorithm in the optimization phase
    opt_tol (int): tolerance for kmeans algorithm convergence in the optimization phase
    opt_maxIter (int): maximum number of iterations for the kmeans algorithm in the optimization phase
    n_cores (int): number of cores to use
    log_path (string): where to save optimization stats. Default None (no saving)
    tr_initStep (int): number of different random intializations for the kmeans algorithm in the training phase
    tr_tol (int): tolerance for kmeans algorithm convergence in the training phase
    tr_maxIter (int): maximum number of iterations for the kmeans algorithm in the training phase

    Returns:
    original_data (pyspark.sql.dataframe.DataFrame): the input data frame with an extra \"prediction\" column
    """
    from language_models import w2v_preproc
    from pyspark.ml.clustering import KMeansModel
    import time
    import datetime
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pathlib import Path

    if kmeans_mode not in ["load", "train"]:
        print("""WARNING: invalid param \"kmeans_mode\". Specify either \"load\" to train load a pre-trained model 
              or \"train\" to train it online.""")
        return (None)

    original_data = w2v_preproc(original_data, msg_col, id_col, w2v_model_path)

    if kmeans_mode == "load":
        original_data = kmeans_preproc(original_data, tks_vec)
        kmeans_model = KMeansModel.load(kmeans_model_path)
    else:
        # K_optim()
        # initialize a grid of K (number of clusters) values
        #         k_list = [12, 16, 20]

        # train for different Ks
        res = K_optim(k_list, dataset=original_data, tks_vec=tks_vec, ft_col=ft_col,
                      distance=distance, initSteps=opt_initSteps, tol=opt_tol, maxIter=opt_maxIter,
                      n_cores=n_cores, log_path=log_path)

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

    return (original_data)