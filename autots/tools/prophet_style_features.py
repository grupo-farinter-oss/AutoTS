"""Design-matrix builders shared by FastProphet (Fourier, changepoints, holidays).

Everything here is deterministic given a DatetimeIndex, so the exact same builder is
applied to the training index at fit time and to the future index at predict time.
"""

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0


def index_to_days(index: pd.DatetimeIndex) -> np.ndarray:
    """Float days since the unix epoch (physical time, frequency independent)."""
    return index.to_numpy(dtype='datetime64[ns]').astype('int64') / (
        1e9 * SECONDS_PER_DAY
    )


def fourier_matrix(days: np.ndarray, period_days: float, order: int) -> np.ndarray:
    """[sin(2 pi k t / P), cos(2 pi k t / P)] for k=1..order -> (n, 2*order)."""
    phase = 2.0 * np.pi * np.outer(days / period_days, np.arange(1, order + 1))
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=1)


def auto_seasonality_flags(days: np.ndarray) -> dict:
    """Prophet's 'auto' rule: enough history to see the cycle and fine enough sampling."""
    span = days[-1] - days[0]
    steps = np.diff(days)
    min_step = steps[steps > 0].min() if (steps > 0).any() else np.inf
    return {
        'yearly': bool(span >= 730),
        'weekly': bool(span >= 14 and min_step < 7),
        'daily': bool(span >= 2 and min_step < 1),
    }


def changepoint_basis(
    days: np.ndarray,
    scaled_time: np.ndarray,
    history_days: np.ndarray,
    n_changepoints: int,
    changepoint_range: float,
    changepoint_spacing=None,
    changepoint_distance_end=None,
) -> np.ndarray:
    """Return changepoint locations (in scaled time) for the trend hinge basis.

    Two styles, mirroring FBProphet in this package:
      - classic: ``n_changepoints`` evenly spaced over the first ``changepoint_range``
        fraction of history rows
      - calendar grid (range > 1 or distance_end given): a uniform grid in days
        walking backwards from ``distance_end`` days before the last observation
    """
    n_hist = len(history_days)
    time_hist = scaled_time[:n_hist]
    if changepoint_range > 1 or changepoint_distance_end is not None:
        distance_end = int(
            changepoint_range
            if changepoint_distance_end is None
            else changepoint_distance_end
        )
        spacing = int(
            max(n_hist / max(n_changepoints, 1), 1)
            if changepoint_spacing is None
            else changepoint_spacing
        )
        spacing = max(spacing, 1)
        last = history_days[-1] - distance_end
        candidates = np.arange(last, history_days[0], -spacing)[::-1]
        candidates = candidates[(candidates > history_days[0]) & (candidates < history_days[-1])]
        denominator = history_days[-1] - history_days[0]
        return (candidates - history_days[0]) / denominator if denominator else candidates
    usable = max(int(np.ceil(n_hist * min(changepoint_range, 1.0))), 2)
    count = min(n_changepoints, usable - 1)
    if count <= 0:
        return np.zeros(0)
    positions = np.linspace(0, usable - 1, count + 1).round().astype(int)[1:]
    return time_hist[np.unique(positions)]


def trend_design(scaled_time: np.ndarray, changepoints: np.ndarray) -> np.ndarray:
    """[1, t, (t - s_j)+ ...] -> piecewise linear continuous trend basis."""
    hinge = np.maximum(scaled_time[:, None] - changepoints[None, :], 0.0)
    return np.concatenate(
        [np.ones((len(scaled_time), 1)), scaled_time[:, None], hinge], axis=1
    )


def country_holiday_events(countries, years) -> pd.DataFrame:
    """Long frame (date, name) of country holidays; falls back to one pooled flag."""
    rows = []
    try:
        import holidays as holidays_lib

        for country in countries:
            try:
                known = holidays_lib.country_holidays(str(country), years=list(years))
            except Exception:
                continue
            rows.extend((pd.Timestamp(day), str(name)) for day, name in known.items())
    except ImportError:
        pass
    return pd.DataFrame(rows, columns=['date', 'name'])


def expand_holiday_windows(events: pd.DataFrame) -> pd.DataFrame:
    """Apply optional lower_window/upper_window columns (days), Prophet naming style."""
    if 'lower_window' not in events and 'upper_window' not in events:
        return events
    lower = events.get('lower_window', pd.Series(0, index=events.index)).fillna(0)
    upper = events.get('upper_window', pd.Series(0, index=events.index)).fillna(0)
    pieces = []
    for offset in range(int(lower.min()), int(upper.max()) + 1):
        keep = (lower <= offset) & (upper >= offset)
        if not keep.any():
            continue
        piece = events[keep].copy()
        piece['date'] = piece['date'] + pd.Timedelta(days=offset)
        if offset != 0:
            piece['name'] = piece['name'].astype(str) + f"_{offset:+d}"
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else events.iloc[0:0]


class HolidayDesign:
    """Per-day holiday indicator columns, global or per series.

    Args:
        events: long frame with ``date``, ``name`` and optionally ``series``
            (rows with null/absent series apply to every series).
        series_names: column names of the training frame, defines series index.
        names: fixed column ordering (given at predict time so future matches fit).
    """

    def __init__(self, events: pd.DataFrame, series_names, names=None):
        events = events.dropna(subset=['date', 'name']).copy()
        events['date'] = pd.to_datetime(events['date']).dt.normalize()
        if 'series' not in events:
            events['series'] = np.nan
        self.names = (
            sorted(events['name'].astype(str).unique()) if names is None else list(names)
        )
        events = events[events['name'].astype(str).isin(self.names)]
        self._events_date = events['date'].to_numpy(dtype='datetime64[ns]')
        name_code = {n: i for i, n in enumerate(self.names)}
        self._events_col = events['name'].astype(str).map(name_code).to_numpy(dtype=int)
        series_code = {s: i for i, s in enumerate(series_names)}
        mapped = events['series'].map(series_code)
        self._events_series = mapped.fillna(-1).to_numpy(dtype=int)
        # drop events whose series is not in the frame (but keep explicit globals)
        valid = (self._events_series >= 0) | events['series'].isna().to_numpy()
        self._events_date = self._events_date[valid]
        self._events_col = self._events_col[valid]
        self._events_series = self._events_series[valid]
        self.is_global = bool((self._events_series < 0).all())

    def restrict_to_active(self, index: pd.DatetimeIndex) -> 'HolidayDesign':
        """Drop columns with zero occurrences on ``index`` (they cannot be learned)."""
        days = index.normalize().to_numpy(dtype='datetime64[ns]')
        hit = np.isin(self._events_date, days)
        active = np.unique(self._events_col[hit])
        remap = {old: new for new, old in enumerate(active)}
        keep = np.isin(self._events_col, active)
        self.names = [self.names[i] for i in active]
        self._events_date = self._events_date[keep]
        self._events_col = np.array([remap[c] for c in self._events_col[keep]], dtype=int)
        self._events_series = self._events_series[keep]
        return self

    def dense(self, index: pd.DatetimeIndex, series_slice: slice = None):
        """(n, H) if global else (c, n, H), for rows of ``index`` (float64)."""
        n_cols = len(self.names)
        days = index.normalize().to_numpy(dtype='datetime64[ns]')
        unique_days, inverse = np.unique(days, return_inverse=True)
        day_pos = np.searchsorted(unique_days, self._events_date)
        in_range = (day_pos < len(unique_days)) & (
            unique_days[np.minimum(day_pos, len(unique_days) - 1)] == self._events_date
        )
        if self.is_global:
            by_day = np.zeros((len(unique_days), n_cols))
            by_day[day_pos[in_range], self._events_col[in_range]] = 1.0
            return by_day[inverse]
        start = series_slice.start or 0
        stop = series_slice.stop
        n_chunk = stop - start
        by_day = np.zeros((n_chunk, len(unique_days), n_cols))
        series_ids = self._events_series
        global_rows = in_range & (series_ids < 0)
        by_day[:, day_pos[global_rows], self._events_col[global_rows]] = 1.0
        local = in_range & (series_ids >= start) & (series_ids < stop)
        by_day[series_ids[local] - start, day_pos[local], self._events_col[local]] = 1.0
        return by_day[:, inverse, :]
