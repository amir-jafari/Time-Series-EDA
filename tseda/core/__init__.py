"""
tseda.core
==========

Core data structures and validation utilities.

Public API
----------
TimeSeries
    Univariate time series with a DatetimeIndex.
ArrayLike
    Type alias for 1-D numeric inputs.
DatetimeLike
    Type alias for datetime-index inputs.
Frequency
    Enum of recognised pandas offset aliases.
AggMethod
    Enum of aggregation methods for resampling / rolling.
DiffMethod
    Enum of differencing strategies.
"""
from tseda.core.timeseries import TimeSeries
from tseda.core.types import AggMethod, ArrayLike, DatetimeLike, DiffMethod, Frequency
from tseda.core.validator import (
    validate_data_array,
    validate_datetime_index,
    validate_freq_string,
    validate_lags,
    validate_positive_int,
)

__all__ = [
    # Data structure
    "TimeSeries",
    # Type aliases
    "ArrayLike",
    "DatetimeLike",
    # Enums
    "Frequency",
    "AggMethod",
    "DiffMethod",
    # Validators (useful for extension modules)
    "validate_data_array",
    "validate_datetime_index",
    "validate_freq_string",
    "validate_lags",
    "validate_positive_int",
]