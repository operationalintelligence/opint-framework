from opint_framework.core.nlp.nlp import Vectorization


class LucaVectorization(Vectorization):

    def __init__(self, ctx):
        super(LucaVectorization, self).__init__(ctx)

    def train_model(self, messages, path_to_model, embedding_size=300, window=7, min_count=1, workers=1,
                    training_algorithm=1, iter=10):
        pass

    def load_model(self, path_to_model):
        pass

    def update_model(self, path_to_model, tokenized):
        pass

    def vectorize_messages(self, word2vec, tokenized):
        pass
