import six

from filters.schema import base_query_params_schema
from filters.validations import (CSVofIntegers, IntegerLike)


# make a validation schema for issues filter query params
issue_query_schema = base_query_params_schema.extend(
    {
        "id": IntegerLike(),
        "message": six.text_type,
        "categories": CSVofIntegers(),
    })
