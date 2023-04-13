from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisticMatrixFactorization
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
from typing import List
from typing import Dict, Any
import pandas as pd

MODEL = {
    "lmf": LogisticMatrixFactorization,
    "als": AlternatingLeastSquares,
    "bpr": BayesianPersonalizedRanking,
}


def _get_sparse_matrix(values):
        return csr_matrix(values)


def _get_model(name: str, **params):
    model = MODEL.get(name)
    if model is None:
        raise ValueError("No model with name {}".format(name))
    return model(**params)


class InternalStatusError(Exception):
    pass

def get_user_row_index(user_id: int, user_index_mapping: pd.Series) -> int:
    return user_index_mapping[user_id]

class Recommender:
    def __init__(
        self,
        values,
    ):
        self.user_product_matrix = _get_sparse_matrix(values)

        # This variable will be set during the training phase
        self.model = None
        self.fitted = False


    def create_and_fit(
        self,
        model_name: str,
        weight_strategy: str = "bm25",
        model_params: Dict[str, Any] = {},
    ):
        weight_strategy = weight_strategy.lower()
        if weight_strategy == "bm25":
            data = bm25_weight(
                self.user_product_matrix,
                K1=1.2,
                B=0.75,
            )
        elif weight_strategy == "balanced":
            # Balance the positive and negative (nan) entries
            # http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf
            total_size = (
                self.user_product_matrix.shape[0] * self.user_product_matrix.shape[1]
            )
            sum = self.user_product_matrix.sum()
            num_zeros = total_size - self.user_product_matrix.count_nonzero()
            data = self.user_product_matrix.multiply(num_zeros / sum)
        elif weight_strategy == "same":
            data = self.user_product_matrix
        else:
            raise ValueError("Weight strategy not supported")

        self.model = _get_model(model_name, **model_params)
        self.fitted = True

        self.model.fit(data)

        return self

    def recommend_products(self, user_id: int, n: int = 10) -> List[int]:
        """
        Recommends n products to the user with id user_id.
        """
        if not self.fitted:
            raise InternalStatusError(
                "Cannot recommend products without previously fitting the model."
                " Please, consider fitting the model before recommening products."
            )

        scores = self.model.recommend(user_id, self.user_product_matrix, N=n, filter_already_liked_items=True)
        suggestions = [s[0] for s in scores]
        return suggestions, scores

        return self.model.recommend(
            user_id,
            self.user_product_matrix[user_id],
            filter_already_liked_items=True,
            N=items_to_recommend,
        )

    def explain_recommendation(
        self,
        user_id,
        suggested_item_id,
        recommended_items,
    ):
        _, items_score_contrib, _ = self.model.explain(
            user_id,
            self.user_product_matrix,
            suggested_item_id,
            N=recommended_items,
        )

        return items_score_contrib

    def similar_users(self, user_id):
        return self.model.similar_users(user_id)

    @property
    def item_factors(self):
        return self.model.item_factors
