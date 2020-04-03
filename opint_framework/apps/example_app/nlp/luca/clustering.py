from opint_framework.core.nlp.nlp import Clustering
from opint_framework.apps.example_app.nlp.luca.kmeans import *

class LucaClustering(Clustering):

    def update_model(self, path_to_model, tokenized):
        pass

    def __init__(self, ctx):
        super(LucaClustering, self).__init__(ctx)

    def data_preparation(self, messages, tks_vec):
        """Take input dataset with Word2Vec representation in tks_vec column and properly format to feed into pyspark.ml.KMeans."""
        from pyspark.ml.feature import VectorAssembler
        vec_assembler = VectorAssembler(inputCols=[tks_vec], outputCol='features')
        messages = vec_assembler.transform(messages)
        return (messages)

    def train_model(self, messages, k, path_to_model=None, ft_col='features', distance="cosine",
                    initSteps=10, tol=0.0001, maxIter=30, mode="new", log_path=None):
        """Train K-Means model.

        -- params:
        messages (pyspark.sql.dataframe.DataFrame): data frame with a vector column with features for the kmeans algorithm
        k (int): number of clusters
        ft_col (string): name of the features column
        distance ("euclidean" or "cosine"): distance measure for the kmeans algorithm
        tol (int): tolerance for kmeans algorithm convergence
        maxIter (int): maximum number of iterations for the kmeans algorithm
        path_to_model (string): where to save trained kmeans model
        mode ("new" or "overwrite"): whether to save new file or overwrite pre-existing one.
        log_path (string): where to save optimization stats. Default None (no saving)

        Returns:
        model_fit (pyspark.ml.clustering.KMeansModel): trained K-Means model
        """
        from pyspark.ml.evaluation import ClusteringEvaluator
        from pyspark.ml.clustering import KMeans
        from pathlib import Path
        import time
        import datetime

        evaluator = ClusteringEvaluator(distanceMeasure=distance)

        start_time = time.time()
        start_time_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

        model = KMeans(featuresCol=ft_col, k=k, initMode='k-means||',
                       initSteps=initSteps, tol=tol, maxIter=maxIter, distanceMeasure=distance)

        model_fit = model.fit(messages)

        wsse = model_fit.summary.trainingCost
        silhouette = evaluator.evaluate(model_fit.summary.predictions)

        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print("Saving training metrics to: {}".format(log_path))
            with open(log_path, "a") as log:
                log.write("With K={}\n\n".format(k))
                log.write("Started at: {}\n".format(start_time_string))
                log.write("Within Cluster Sum of Squared Errors = " + str(round(wsse, 4)))
                log.write("\nSilhouette with cosine distance = " + str(round(silhouette, 4)))

                log.write("\nTime elapsed: {} minutes and {} seconds.\n".format(int((time.time() - start_time) / 60),
                                                                                int((time.time() - start_time) % 60)))
                log.write('--' * 30 + "\n\n")
        else:
            print("With K={}\n".format(k))
            print("Started at: {}\n".format(start_time_string))
            print("Within Cluster Sum of Squared Errors = " + str(round(wsse, 4)))
            print("Silhouette with cosine distance = " + str(round(silhouette, 4)))

            print("\nTime elapsed: {} minutes and {} seconds.".format(int((time.time() - start_time) / 60),
                                                                      int((time.time() - start_time) % 60)))
            print('--' * 30)

        if path_to_model:
            outname = "{}/kmeans_K={}".format(path_to_model, k)
            print("Saving K-Means model to: {}".format(outname))
            if mode == "overwrite":
                model.write().overwrite().save(outname)
            else:
                model.save(outname)

        return {"model": model_fit, "wsse": wsse, "asw": silhouette}

    def load_model(self, path_to_model):
        """Load K-Means model from path_to_model."""
        from pyspark.ml.clustering import KMeansModel
        model = KMeansModel.load(path_to_model)
        return (model)

    def K_optim(self, k_list, messages, tks_vec="message_vector", ft_col="features", distance="cosine",
                initSteps=10, tol=0.0001, maxIter=30, n_cores=8, log_path=None):
        """Train K-Means model for different K values.

        -- params:
        k_list (list): grid of K values to try
        messages (pyspark.sql.dataframe.DataFrame): data frame with a vector column with features for the kmeans algorithm
        tks_vec (string): name of the word2vec representations column
        ft_col (string): name of the features column
        distance ("euclidean" or "cosine"): distance measure for the kmeans algorithm
        initStep (int): number of different random intializations for the kmeans algorithm
        tol (int): tolerance for kmeans algorithm convergence
        maxIter (int): maximum number of iterations for the kmeans algorithm
        n_cores (int): number of cores to use
        log_path (string): where to save optimization stats. Default None (no saving)

        Returns:
        res (dict): dictionary with grid of trained models and evaluation metrics. Keys:{"model", "wsse", "silhouette"}
        """
        import time
        import datetime
        from pathlib import Path
        from multiprocessing.pool import ThreadPool

        messages = self.data_preparation(messages, tks_vec) #kmeans_preproc(messages, tks_vec)

        if n_cores > 1:
            pool = ThreadPool(n_cores)
            models_k = pool.map(lambda k: self.train_model(messages, ft_col=ft_col, k=k, distance=distance,
                                                       initSteps=initSteps, tol=tol, maxIter=maxIter,
                                                       log_path=log_path), k_list)
            clustering_models = [k_dict["model"] for k_dict in models_k]
            wsse = [k_dict["wsse"] for k_dict in models_k]
            silhouette = [k_dict["asw"] for k_dict in models_k]
        else:
            clustering_models = []
            wsse = []
            silhouette = []
            for i, k in enumerate(k_list):
                start_time = time.time()
                start_time_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

                print("Training for K={}".format(k))
                print("Started at: {}\n".format(start_time_string))

                model_k = self.clusterization.train_kmeans(messages, ft_col=ft_col, k=k, distance="cosine",
                                       initSteps=initSteps, tol=tol, maxIter=maxIter, log_path=log_path)

                print("\nTime elapsed: {} minutes and {} seconds.".format(int((time.time() - start_time) / 60),
                                                                          int((time.time() - start_time) % 60)))
                print('--' * 30)

                # compute metrics
                clustering_models.append(model_k["model"])
                wsse.append(model_k["wsse"])
                silhouette.append(model_k["asw"])

        res = {"model": clustering_models, "wsse": wsse, "silhouette": silhouette}
        return (res)

    def predict(self, model, tokenized, pred_mode="static", new_cluster_thresh=None, update_model_path=None):
        """Predict cluster for new observations.

        -- params:
        messages (pyspark.sql.dataframe.DataFrame): data frame with a vector column with features for the kmeans algorithm
        model (pyspark.ml.clustering.KMeansModel): re-trained kmeans model
        pred_mode ("static" or "update"): prediction mode: "static" does not allow for creating new clusters
        distance ("euclidean" or "cosine"): distance measure for the kmeans algorithm
        new_cluster_thresh (float): distance threshold: if closest centroid is more distant than new_cluster_thresh
                                    then a new cluster is created for the new observation
        update_model_path (string): where to save update kmeans model

        Returns:
        pred (pyspark.sql.dataframe.DataFrame): the input data frame with an extra "prediction" column
        """
        if pred_mode not in ["static", "update"]:
            print("""WARNING: invalid param \"pred_mode\". Specify either \"static\" to train load a pre-trained model 
                  or \"update\" to train it online.""")
            return (None)
        if pred_mode == "static":
            pred = model.transform(tokenized)
        else:
            # take centroids
            # compute distances of each message from each centroid
            # select closest centroid per each meassage
            # initialize new clusters when closest centroid distance is greater than new_cluster_thresh
            # update centroids and points in each cluster
            # save updated model
            update_model_path = "temp_filename"  # temporary to avoid accidental overwriting
            model.write().overwrite().save(update_model_path)
            pred = None

        return (pred)