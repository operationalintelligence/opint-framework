def stats_summary(dataset, clust_col="prediction", tks_col="stop_token_1", abs_tks_out="abstract_message",
                  abstract=True):
    """Compute frequencies of unique messages aggregated per cluster.

    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with tokens lists and cluster prediction columns
    clust_col (string): name of the cluster prediction column
    tks_col (string): name of the tokens lists column
    abs_tks_out (string): name of the output column for abstract tokens if abstract is True
    abstract (bool): whether to consider abstract tokens when selecting unique messages

    Returns:
    stats_summary (pandas.DataFrame): data frame with:
                                    "n_messages" --> number of messages per cluster
                                    "unique_strings" --> number of unique messages per cluster
                                    "unique_patterns" (if abstract==True) --> number of unique abstract messages per cluster
    """
    import pyspark.sql.functions as F

    grouped_stats = dataset.groupBy(clust_col)
    if abstract:
        stats_summary = grouped_stats.agg(F.count(tks_col).alias("n_messages"),
                                          F.countDistinct(tks_col).alias("unique_strings"),
                                          F.countDistinct(abs_tks_out).alias("unique_patterns"),
                                          ).orderBy("n_messages", ascending=False)
        # add column
    else:
        stats_summary = grouped_stats.agg(F.count(tks_col).alias("n_messages"),
                                          F.countDistinct(tks_col).alias("unique_strings"),
                                          ).orderBy("n_messages", ascending=False)

    return (stats_summary)  # .toPandas())


def pattern_summary(dataset, clust_col="prediction", tks_col="stop_token_1", abs_tks_out="abstract_tokens",
                    abstract=True, n_mess=3, original=None, n_src=3, n_dst=3,
                    src_col=None, dst_col=None, save_path=None, tokenization=None):
    """Compute top n_mess messages aggregated per cluster.

    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with tokens lists and cluster prediction columns
    clust_col (string): name of the cluster prediction column
    tks_col (string): name of the tokens lists column
    abs_tks_in (string): name of the column with tokens to be abstracted if abstract is True
    abs_tks_out (string): name of the output column for abstract tokens if abstract is True
    abstract (bool): whether to consider abstract tokens when selecting unique messages
    n_mess (int): number of most frequent patterns to retain
    original (pyspark.sql.dataframe.DataFrame): data frame with hdfs data for enriched summary.
                                                Default None (no additional information is showed)
    n_src (int): number of most frequent source sites to retain  -- Default None (TO DO)
    n_dst (int): number of most frequent destination sites to retain  -- Default None (TO DO)
    src_col (string): name of the source site column in the original data frame  -- Default None (TO DO)
    dst_col (string): name of the destination site column in the original data frame  -- Default None (TO DO)

    Returns:
    patterns_summary (pandas.DataFrame): data frame with:
                                    "top_{n_mess}" --> dictionary with top n_mess patterns per cluster
                                            (Keys are ["msg": contains the pattern, "n": relative frequency in the cluster]
                                    "top_{n_src}" --> dictionary with top n_src source sites per cluster
                                            (Keys are ["src": contains the source, "n": relative frequency in the cluster]
                                    "top_{n_dst}" --> dictionary with top n_mess per cluster
                                            (Keys are ["dst": contains the destination, "n": relative frequency in the cluster]
    """
    import pandas as pd
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window

    # extract top N patterns per each cluster
    if abstract:
        msg_col = (abs_tks_out, "n_patterns")
    else:
        msg_col = (tks_col, "n_strings")

    grouped_patterns_msg = stats_pattern(dataset=dataset, clust_col=clust_col,
                                         agg_col_in=msg_col[0], agg_col_out=msg_col[1], n_rank=n_mess,
                                         save_path=save_path, tokenization=tokenization)
    #
    if original:
        grouped_patterns_src = stats_pattern(dataset=dataset, clust_col=clust_col,
                                             agg_col_in=src_col, agg_col_out="n_src", n_rank=n_src,
                                             save_path=save_path)

        grouped_patterns_dst = stats_pattern(dataset=dataset, clust_col=clust_col,
                                             agg_col_in=dst_col, agg_col_out="n_dst", n_rank=n_dst,
                                             save_path=save_path)

    # merge summary data frames
    summary_table = grouped_patterns_msg.join(grouped_patterns_src,
                                              on=[clust_col, "rank_pattern"], how="full")
    summary_table = summary_table.join(grouped_patterns_dst,
                                       on=[clust_col, "rank_pattern"], how="full")
    return (summary_table)


def summary(dataset, k=None, clust_col="prediction", tks_col="stop_token_1", abs_tks_out="abstract_message",
            abs_tks_in="tokens_cleaned", abstract=True, n_mess=3, wrdcld=False,  # stats_summary
            original=None, n_src=3, n_dst=3, src_col=None, dst_col=None, data_id="tr_id", orig_id="tr_id",
            # patterns_summary
            save_path=None, timeplot=False, time_col=None, tokenization=None, model_ref=None
            ):
    """Return summary statistics aggregated per cluster.

    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with tokens lists and cluster prediction columns
    k (int): number of clusters. If specified the executing time decreases. Default None
    clust_col (string): name of the cluster prediction column
    tks_col (string): name of the tokens lists column
    abs_tks_in (string): name of the column with tokens to be abstracted if abstract is True
    abs_tks_out (string): name of the output column for abstract tokens if abstract is True
    abstract (bool): whether to consider abstract tokens when selecting unique messages
    n_mess (int): number of most frequent patterns to retain
    wrdcld (bool): whether to produce word cloud fr visualization of clusters content
    original (pyspark.sql.dataframe.DataFrame): data frame with hdfs data for enriched summary.
                                                Default None (no additional information is showed)
    n_src (int): number of most frequent source sites to retain  -- Default None (TO DO)
    n_src (int): number of most frequent destination sites to retain  -- Default None (TO DO)
    src_col (string): name of the source site column in the original data frame  -- Default None (TO DO)
    dst_col (string): name of the destination site column in the original data frame  -- Default None (TO DO)
    data_id (string): name of the message id column in the dataset data frame
    orig_id (string): name of the message id column in the original data frame
    save_path (string): base folder path where to store outputs
    timeplot (bool): whether to plot errors time trend
    time_col (string): name of the unix time in milliseconds

    Returns:
    summary_df (pandas.DataFrame): merged data frame with stats_summary and patterns_summary
    """
    import pandas as pd
    import pyspark.sql.functions as F
    from pyspark.sql.types import StringType
    # from opint_framework.apps.example_app.nlp.pyspark_based.tokenization import abstract_params
    from pathlib import Path

    # rename tokens and  detokenized abstract column in the output
    out_tks_col = "tokens"
    detoken_out_abs_col = "pattern"

    pd.options.mode.chained_assignment = 'raise'
    # compute quantitative stats of the clusters
    if abstract:
        dataset = tokenization.abstract_params(dataset, tks_col=abs_tks_in, out_col=abs_tks_out)
        dataset = tokenization.detokenize_messages(dataset, abs_tks_out, detoken_out_abs_col)

    if original:
        or_cols = original.columns
        data_cols = [dataset[col] for col in dataset.columns if col not in or_cols]

        out_cols = [original[col] for col in or_cols]
        out_cols.extend(data_cols)

        dataset = original.join(dataset, original[orig_id].alias("id") == dataset[data_id],
                                how="outer").select(out_cols)
        dataset = convert_endpoint_to_site(dataset, "src_hostname", "dst_hostname")

    if timeplot:
        plot_time(dataset, time_col=time_col, clust_col=clust_col, k=k, save_path="{}/plots/timeplot".format(save_path))

    # tokens cloud
    if wrdcld:
        tokens_cloud(dataset, msg_col=abs_tks_out, clust_col=clust_col, save_path="{}/plots/token_clouds".format(save_path))


    # add model reference column and UUID for cluster label
    import uuid
    UUID_dict = {}

    if k is None:
        k = dataset.prediction.countDistinct()

    for i in range(k):
        UUID_dict[i] = str(uuid.uuid4())

    def uuid_str(clust_label, UUID_dict=UUID_dict):
        uuid = UUID_dict.get(clust_label, None)
        return (uuid)

    model_ref_udf = F.udf(lambda: model_ref, StringType())
    uuid_udf = F.udf(uuid_str, StringType())
    dataset = dataset.withColumn("model_ref", model_ref_udf()).withColumn("clust_UUID", uuid_udf(clust_col))

    # save raw prediction dataset
    import datetime
    date_hdfs_format = str(datetime.date.today()).replace("-", "/")
    save_path = save_path + "/{}".format(date_hdfs_format)

    print("Saving raw prediction dataset to: {}/raw".format(save_path))

    if abstract:
        dataset = dataset.select(data_id, "t__error_message", F.col(tks_col).alias(out_tks_col), detoken_out_abs_col, src_col, dst_col, time_col,
                                 clust_col, "clust_UUID", "model_ref")
    else:
        dataset = dataset.select(data_id, "t__error_message", F.col(tks_col).alias(out_tks_col), src_col, dst_col, time_col,
                                 clust_col, "clust_UUID", "model_ref")
    dataset.write.format('json').mode('overwrite').save("{}/raw".format(save_path))

    # first compute quantitative stats of the clusters
    stats = stats_summary(dataset, clust_col=clust_col, tks_col=out_tks_col, abs_tks_out=detoken_out_abs_col, abstract=abstract)

    # second extract top N most frequent patterns
    patterns = pattern_summary(dataset, clust_col=clust_col, tks_col=out_tks_col, abs_tks_out=detoken_out_abs_col,
                               abstract=abstract, n_mess=n_mess, original=original,
                               n_src=n_src, n_dst=n_dst, src_col=src_col, dst_col=dst_col, save_path=save_path)

    summary_df = stats.join(patterns, on=clust_col, how="full").orderBy(F.col("n_messages").desc(),
                                                                        clust_col, "rank_pattern")

    # add model reference column and UUID for cluster label
    summary_df = summary_df.withColumn("model_ref", model_ref_udf()).withColumn("clust_UUID", uuid_udf(clust_col))

    # reorder columns
    cols = [summary_df.columns[-1]] + summary_df.columns[:-1]
    summary_df = summary_df.select(cols)

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        outname = save_path / "aggregate/summary.json"  # .format(clust_id[clust_col])
        print("Saving clustering summary to: {}".format(outname))
        save_to_json(summary_df, save_path=outname)

    # transform to pandas
    summary_df = summary_df.toPandas().set_index([clust_col, "rank_pattern"])


    return (dataset, summary_df)


def stats_pattern(dataset, clust_col, agg_col_in, agg_col_out, n_rank, save_path=None, tokenization=None):
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window

    if "src" in agg_col_in:
        agg_col_label = "src"
    elif "dst" in agg_col_in:
        agg_col_label = "dst"
    else:
        agg_col_label = "msg"

    grouped_patterns = dataset.groupBy(clust_col, agg_col_in).agg(F.count("*").alias(agg_col_out)
                                                                  ).orderBy(clust_col, agg_col_out,
                                                                            ascending=[True, False])
    window_pattern = Window.partitionBy(clust_col).orderBy(F.col(agg_col_out).desc(),
                                                           F.col(agg_col_in))
    window_cluster = Window.partitionBy(clust_col).orderBy(F.col(clust_col))
    if n_rank:
        grouped_patterns = grouped_patterns.select('*',
                                                   (F.col(agg_col_out) / F.sum(F.col(agg_col_out)).over(
                                                       window_cluster)).alias("{}_perc".format(agg_col_label)),
                                                   F.rank().over(window_pattern).alias('rank_pattern')) \
            .filter(F.col('rank_pattern') <= n_rank)
    else:
        grouped_patterns = grouped_patterns.select('*',
                                                   (F.col(agg_col_out) / F.sum(F.col(agg_col_out)).over(
                                                       window_cluster)).alias("{}_perc".format(agg_col_label)),
                                                   F.rank().over(window_pattern).alias('rank_pattern'))
    if tokenization:
        cols = grouped_patterns.columns
        cols[cols.index(agg_col_in)] = "pattern"
        grouped_patterns = tokenization.detokenize_messages(grouped_patterns, agg_col_in)
        grouped_patterns = grouped_patterns.drop(agg_col_in)
        grouped_patterns = grouped_patterns.withColumnRenamed("message_string", "pattern").select(cols)

    if save_path:
        outname = "{}/aggregate/{}_aggregate_summary.json".format(save_path, agg_col_label)
        print("Saving {} aggregate summary to: {}".format(agg_col_label, outname))
        save_to_json(grouped_patterns, save_path=outname)

    return (grouped_patterns)


def save_to_json(summary_table, save_path):
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w+") as outfile:
        outfile.write("[\n")
        # print(summary_table.toPandas().head())
        for i, row in enumerate(summary_table.collect()):
            outfile.write("{")
            for col in summary_table.columns:
                if col == summary_table.columns[-1]:
                    if type(row[col]) is not str:
                        outfile.write("\"{}\":{}".format(col, row[col]))
                    else:
                        outfile.write("\"{}\":\"{}\"".format(col, row[col]))
                    outfile.write("},\n")

                else:
                    if type(row[col]) is not str:
                        outfile.write("\"{}\":{},".format(col, row[col]))
                    else:
                        outfile.write("\"{}\":\"{}\",".format(col, row[col]))

        # remove last comma and close the initial square bracket
        outfile.seek(outfile.tell() - 2, 0)
        outfile.truncate()
        outfile.write("\n]")


def tokens_cloud(dataset, msg_col, clust_col="prediction", save_path=None,
                 figsize=(8, 4), width=800, height=400, bkg_col="white", min_font_size=11):
    """Return summary statistics aggregated per cluster.

    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with tokens lists and cluster prediction columns
    msg_col (string): name of the tokens lists column
    clust_col (string): name of the cluster prediction column
    save_path (string): where to save output figures. Defaule None (no saving)
    figsize (tuple(int, int)): figure size
    width (int): width of word clouds
    height (int): height of word clouds
    bkg_col (string): background color of word clouds
    min_font_size (int): fontsize for the least commond tokens

    Returns: None
    """
    import os
    import wordcloud as wrdcld
    import matplotlib
    import pyspark.sql.functions as F
    from matplotlib import pyplot as plt
    # from opint_framework.apps.example_app.nlp.pyspark_based.tokenization import abstract_params
    from pathlib import Path

    if save_path:
        print("Saving tokens cloud to: {}".format(save_path))

    for clust_id in dataset.select(clust_col).distinct().collect():
        cluster_messages = dataset.filter(F.col(clust_col) == clust_id[clust_col]).select(msg_col).collect()
        if type(cluster_messages[0][msg_col]) == type([]):
            cluster_messages = [tkn.strip() for tks_msg in cluster_messages for tkn in tks_msg[msg_col]]
        else:
            cluster_messages = [msg[msg_col].strip() for msg in cluster_messages]

        # Create and generate a word cloud image:
        wordcloud = wrdcld.WordCloud(width=width, height=height, background_color=bkg_col,
                                     min_font_size=min_font_size,
                                     colormap=matplotlib.cm.inferno).generate(" ".join(cluster_messages))

        # Display the generated image:
        fig = plt.figure(figsize=figsize)
        plt.title("CLUSTER {}".format(clust_id[clust_col]))
        plt.axis("off")
        plt.imshow(wordcloud, interpolation='bilinear')
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            outname = save_path / "cluster_{}.png".format(clust_id[clust_col])
            # print("Saving token clouds to: {}".format(outname))
            if os.path.isfile(outname):
                os.remove(outname)
            fig.savefig(outname, format='png', bbox_inches='tight')


def get_hostname(endpoint):
    """
    Extract hostname from the endpoint.
    Returns empty string if failed to extract.

    :return: hostname value
    """
    import re
    p = r'^(.*?://)?(?P<host>[\w.-]+).*'
    r = re.search(p, endpoint)

    return r.group('host') if r else ''


def convert_endpoint_to_site(dataset, src_col, dst_col):
    """
    Convert src/dst hostname to the respective site names.

    :return: dataset
    """
    import requests  # , re
    from pyspark.sql.functions import col, create_map, lit
    from itertools import chain

    # retrieve mapping
    cric_url = "http://wlcg-cric.cern.ch/api/core/service/query/?json&type=SE"
    r = requests.get(url=cric_url).json()
    site_protocols = {}
    for site, info in r.items():
        if "protocols" in info:
            # print(se, type(se), info, type(info))
            for name, prot in info.get('protocols', {}).items():
                site_protocols.setdefault(get_hostname(prot['endpoint']), site)

    # apply mapping
    mapping_expr = create_map([lit(x) for x in chain(*site_protocols.items())])
    out_cols = dataset.columns
    dataset = dataset.withColumnRenamed(src_col, "src")
    dataset = dataset.withColumnRenamed(dst_col, "dst")
    dataset = dataset.withColumn(src_col, mapping_expr[dataset["src"]]) \
        .withColumn(dst_col, mapping_expr[dataset["dst"]])
    return (dataset.select(out_cols))


def plot_time(dataset, time_col, clust_col="prediction", k=None, save_path=None):
    """ Plot the trend of error messages over time (per each cluster).

    -- params:
    dataset (pyspark.sql.dataframe.DataFrame): data frame with predictions and message times
    time_col (string): name of the unix time in milliseconds
    clust_col (string): name of the cluster prediction column
    k (int): number of clusters. If specified the executing time decreases. Default None
    save_path (string): where to save output figures. Default None (no saving)

    Returns: None
    """
    import pyspark.sql.functions as F
    import os
    import datetime
    from pathlib import Path
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.units as munits

    dataset = (dataset.filter(F.col(time_col) > 0)  # ignore null values
               .withColumn("datetime_str", F.from_unixtime(F.col(time_col) / 1000))  # datetime (string)
               .withColumn("datetime", F.to_timestamp(F.col('datetime_str'), 'yyyy-MM-dd HH:mm'))  # datetime (numeric)
               .select(clust_col, "datetime"))
    if k:
        clust_ids = [{"prediction": i} for i in range(0, k)]
    else:
        clust_ids = dataset.select(clust_col).distinct().collect()

    if save_path:
        print("Saving time plots to: {}".format(save_path))

    for clust_id in clust_ids:
        cluster = dataset.filter(F.col(clust_col) == clust_id[clust_col]).select("datetime")
        #         cluster = cluster.groupBy("datetime").agg(F.count("datetime").alias("freq")).orderBy("datetime", ascending=True)
        cluster = cluster.toPandas()

        try:
            res_sort = cluster.datetime.value_counts(bins=24 * 6).sort_index()
        except ValueError:
            print("""WARNING: time column completely empty. Errors time trend 
                  cannot be displayed for cluster {}""".format(clust_id[clust_col]))
            continue

        x_datetime = [interval.right for interval in res_sort.index]

        converter = mdates.ConciseDateConverter()
        munits.registry[datetime.datetime] = converter

        fig, ax = plt.subplots(figsize=(10, 5))
        #         ax.plot(res_sort.index, res_sort)
        ax.plot(x_datetime, res_sort.values)
        min_h = min(x_datetime)  # res_sort.index.min()
        max_h = max(x_datetime)  # res_sort.index.max()
        day_min = str(min_h)[:10]
        day_max = str(max_h)[:10]
        #         title = f"{'Cluster {} - init:'.format(3):>25}{day_min:>15}{str(min_h)[11:]:>12}" + \
        #                  f"\n{'          - end:':>25}{day_max:>15}{str(max_h)[11:]:>12}"
        title = "Cluster {} - day: {}".format(clust_id[clust_col], day_min)
        plt.title(title)

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            outname = save_path / "cluster_{}.png".format(clust_id[clust_col])
            # print("Saving time plots to: {}".format(outname))
            if os.path.isfile(outname):
                os.remove(outname)
            fig.savefig(outname, format='png', bbox_inches='tight')
        else:
            plt.show()
