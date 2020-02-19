
class BaseLoader(object):
    """
        Base data provider (used to load data from external sources)
    """

    @classmethod
    def fetch_data(cls, **kwargs):
        """
        Makes the connection to external source and fetches the data
        To be overwritten by parent

        :return: (data fetched from source)
        """
        return {}

    @classmethod
    def translate_data(cls, data, **kwargs):
        """
        Translates the data from the source format to any desired format.
        Can be overwritten by parent.

        :return: (data translated to desired format)
        """
        return data

    @classmethod
    def load_data(cls, **kwargs):
        """
        :param cache: file path to cache return of loader
        :return: (data loaded from url, etag)
        """

        cache = kwargs.get('cache')
        verbose = kwargs.get('verbose', False)

        try:
            ret = cls.fetch_data(**kwargs)
            data = cls.translate_data(ret, **kwargs)
            if cache:
                with open(cache, "w+") as f:
                    f.write(data)
                if verbose:
                    print('.. saved data into cache file=%s' % cache)
            return data
        except Exception as e:
            print('WARNING: Couldn\'t read data from source. Trying to use cache file. Error: %s' % e)
            try:
                with open(cache, "r") as f:
                    content = f.read()
                    return content
            except Exception as e:
                print('ERROR: cache file=%s is not readable: %s .. skipped' % (cache, e))
