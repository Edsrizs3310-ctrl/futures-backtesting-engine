"""
Data quality validation module.

Performs comprehensive checks on OHLCV data:
    - Missing bar detection (gaps in time series)
    - OHLC consistency (High >= Low, etc.)
    - Volume anomaly detection
    - Data completeness metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Container for validation results."""

    symbol: str
    timeframe: str
    total_bars: int
    missing_bars: int
    ohlc_violations: int
    volume_anomalies: int
    price_anomalies: int
    quality_score: float
    issues: List[str]

    @property
    def is_valid(self) -> bool:
        """Returns True when the dataset passes the quality threshold."""
        return self.quality_score >= 0.95 and self.price_anomalies == 0


class DataValidator:
    """
    Validates OHLCV data quality.

    Methodology:
        The validator operates at two levels. The core ``validate()`` method
        checks one normalized DataFrame, while ``validate_cache_directory()``
        scans parquet caches using the repository's ``SYMBOL_timeframe.parquet``
        naming contract and validates each matching dataset in place.
    """

    SUPPORTED_TIMEFRAMES: Tuple[str, ...] = ("1m", "5m", "30m", "1h")

    GAP_THRESHOLDS = {
        "1m": 30,
        "5m": 30,
        "30m": 180,
        "1h": 180,
    }

    ROLL_JUMP_THRESHOLDS = {
        "1m": 0.05,
        "5m": 0.05,
        "30m": 0.08,
        "1h": 0.08,
    }

    def __init__(self, volume_zscore_threshold: float = 5.0) -> None:
        """
        Initializes the validator.

        Args:
            volume_zscore_threshold: Z-score threshold for volume anomalies.
        """
        self.volume_zscore_threshold = volume_zscore_threshold

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "5m",
    ) -> ValidationResult:
        """
        Runs all validation checks on one DataFrame.

        Args:
            df: OHLCV DataFrame with a datetime index.
            symbol: Symbol name for reporting.
            timeframe: Cache timeframe suffix such as ``5m``.

        Returns:
            ValidationResult with aggregated findings.
        """
        issues: List[str] = []

        if df.empty:
            return ValidationResult(
                symbol=symbol,
                timeframe=timeframe,
                total_bars=0,
                missing_bars=0,
                ohlc_violations=0,
                volume_anomalies=0,
                price_anomalies=0,
                quality_score=0.0,
                issues=["Empty DataFrame"],
            )

        total_bars = len(df)
        max_gap = self.GAP_THRESHOLDS.get(timeframe, 30)

        missing_bars, gap_issues = self._check_gaps(df, max_gap)
        issues.extend(gap_issues)

        ohlc_violations, ohlc_issues = self._check_ohlc_consistency(df)
        issues.extend(ohlc_issues)

        volume_anomalies, volume_issues = self._check_volume_anomalies(df)
        issues.extend(volume_issues)

        price_anomalies, price_issues = self._check_price_continuity(df, timeframe)
        issues.extend(price_issues)

        penalty = (
            missing_bars
            + ohlc_violations * 10
            + volume_anomalies
            + price_anomalies * 25
        ) / max(total_bars, 1)
        quality_score = max(0.0, min(1.0, 1.0 - penalty))

        return ValidationResult(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=total_bars,
            missing_bars=missing_bars,
            ohlc_violations=ohlc_violations,
            volume_anomalies=volume_anomalies,
            price_anomalies=price_anomalies,
            quality_score=quality_score,
            issues=issues,
        )

    def validate_cache_directory(
        self,
        cache_dir: Path,
        symbol: Optional[str] = None,
        timeframes: Optional[Sequence[str]] = None,
    ) -> List[ValidationResult]:
        """
        Validates one symbol or the full cache directory.

        Args:
            cache_dir: Directory containing cached parquet files.
            symbol: Optional symbol filter, for example ``YM``.
            timeframes: Optional timeframe filter such as ``["5m", "1h"]``.

        Returns:
            Validation results sorted by symbol then timeframe.
        """
        requested_symbol = symbol.upper() if symbol else None
        requested_timeframes = self._normalize_timeframes(timeframes)
        results: List[ValidationResult] = []

        for cache_file in sorted(cache_dir.glob("*_*.parquet")):
            parsed = self._parse_cache_filename(cache_file)
            if parsed is None:
                continue

            file_symbol, timeframe = parsed
            if requested_symbol and file_symbol != requested_symbol:
                continue
            if requested_timeframes and timeframe not in requested_timeframes:
                continue

            try:
                df = self._load_cache_frame(cache_file)
            except Exception as exc:
                results.append(
                    ValidationResult(
                        symbol=file_symbol,
                        timeframe=timeframe,
                        total_bars=0,
                        missing_bars=0,
                        ohlc_violations=0,
                        volume_anomalies=0,
                        price_anomalies=0,
                        quality_score=0.0,
                        issues=[f"Failed to read cache file: {exc}"],
                    )
                )
                continue

            results.append(self.validate(df, symbol=file_symbol, timeframe=timeframe))

        return sorted(results, key=lambda result: (result.symbol, result.timeframe))

    def _check_gaps(self, df: pd.DataFrame, max_gap_minutes: int) -> Tuple[int, List[str]]:
        """
        Checks for large gaps in the time series.

        Args:
            df: Normalized OHLCV DataFrame.
            max_gap_minutes: Maximum allowed gap before it is flagged.

        Returns:
            Tuple of ``(gap_count, issue_messages)``.
        """
        if len(df) < 2:
            return 0, []

        issues: List[str] = []
        threshold = timedelta(minutes=max_gap_minutes)
        suspicious_gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        for previous_ts, current_ts in zip(df.index[:-1], df.index[1:]):
            if current_ts - previous_ts <= threshold:
                continue

            # Ignore expected daily session boundaries and weekend rollovers.
            if previous_ts.date() != current_ts.date():
                continue

            suspicious_gaps.append((previous_ts, current_ts))

        gap_count = len(suspicious_gaps)

        if gap_count > 0:
            issues.append(f"Found {gap_count} gaps > {max_gap_minutes} minutes")
            for gap_start, gap_end in suspicious_gaps[:3]:
                issues.append(f"Gap: {gap_start} to {gap_end}")

        return gap_count, issues

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        """
        Checks basic OHLC consistency rules.

        Args:
            df: Normalized OHLCV DataFrame.

        Returns:
            Tuple of ``(violation_count, issue_messages)``.
        """
        issues: List[str] = []
        violations = 0

        normalized_columns = {str(column).lower() for column in df.columns}
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(normalized_columns):
            issues.append("Missing required OHLC columns")
            return 0, issues

        df_check = df.copy()
        df_check.columns = [str(column).lower() for column in df_check.columns]

        high_low_violations = int((df_check["high"] < df_check["low"]).sum())
        if high_low_violations > 0:
            issues.append(f"High < Low violations: {high_low_violations}")
            violations += high_low_violations

        high_violations = int(
            ((df_check["high"] < df_check["open"]) | (df_check["high"] < df_check["close"])).sum()
        )
        if high_violations > 0:
            issues.append(f"High < Open/Close violations: {high_violations}")
            violations += high_violations

        low_violations = int(
            ((df_check["low"] > df_check["open"]) | (df_check["low"] > df_check["close"])).sum()
        )
        if low_violations > 0:
            issues.append(f"Low > Open/Close violations: {low_violations}")
            violations += low_violations

        return violations, issues

    def _check_volume_anomalies(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        """
        Checks for extreme volume outliers using a z-score threshold.

        Args:
            df: Normalized OHLCV DataFrame.

        Returns:
            Tuple of ``(anomaly_count, issue_messages)``.
        """
        normalized_columns = {str(column).lower() for column in df.columns}
        if "volume" not in normalized_columns:
            return 0, []

        issues: List[str] = []
        volume_column = next(column for column in df.columns if str(column).lower() == "volume")
        volume = df[volume_column]

        mean_vol = volume.mean()
        std_vol = volume.std()
        if std_vol == 0 or pd.isna(std_vol):
            return 0, []

        z_scores = np.abs((volume - mean_vol) / std_vol)
        anomalies = int((z_scores > self.volume_zscore_threshold).sum())

        if anomalies > 0:
            issues.append(f"Volume anomalies (z > {self.volume_zscore_threshold}): {anomalies}")

        return anomalies, issues

    def _check_price_continuity(self, df: pd.DataFrame, timeframe: str) -> Tuple[int, List[str]]:
        """
        Checks continuous-series integrity beyond basic OHLC geometry.

        Methodology:
            Continuous futures can pass gap/OHLC checks while still being
            economically broken at contract splice points. This validator
            flags two high-signal failure modes:

            1. ``average`` outside the candle range, which often means OHLC was
               shifted during back-adjustment while ``average`` remained raw.
            2. Large close-to-close jumps exactly where the ``contract`` column
               changes, which points to a failed rollover adjustment.
        """
        issues: List[str] = []
        anomalies = 0

        normalized_columns = {str(column).lower(): column for column in df.columns}
        required_ohlc = {"high", "low", "close"}
        if not required_ohlc.issubset(normalized_columns):
            return 0, []

        high_col = normalized_columns["high"]
        low_col = normalized_columns["low"]
        close_col = normalized_columns["close"]

        average_col = normalized_columns.get("average")
        if average_col is not None:
            average_outside = (df[average_col] < df[low_col]) | (df[average_col] > df[high_col])
            average_violations = int(average_outside.sum())
            if average_violations > 0:
                anomalies += average_violations
                issues.append(f"Average outside candle range: {average_violations}")

        contract_col = normalized_columns.get("contract")
        if contract_col is None or len(df) < 2:
            return anomalies, issues

        roll_jump_threshold = self.ROLL_JUMP_THRESHOLDS.get(timeframe, 0.08)
        close_returns = df[close_col].pct_change().abs()
        contract_change = df[contract_col].astype(str).ne(df[contract_col].astype(str).shift())
        suspicious_rolls = close_returns[contract_change & (close_returns > roll_jump_threshold)].dropna()

        roll_count = int(len(suspicious_rolls))
        if roll_count > 0:
            anomalies += roll_count
            issues.append(
                f"Suspicious roll jumps (> {roll_jump_threshold:.0%} close-to-close): {roll_count}"
            )
            for timestamp, change in suspicious_rolls.head(3).items():
                issues.append(f"Roll jump at {timestamp}: {change:.2%}")

        return anomalies, issues

    def generate_report(self, results: List[ValidationResult]) -> str:
        """
        Generates a human-readable validation report.

        Args:
            results: Validation results to summarize.

        Returns:
            Formatted multi-line report.
        """
        if not results:
            return "\n".join(
                [
                    "=" * 60,
                    "DATA QUALITY REPORT",
                    "=" * 60,
                    "",
                    "No matching cache datasets found.",
                ]
            )

        lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            "",
        ]

        for result in results:
            status = "PASS" if result.is_valid else "FAIL"
            lines.append(
                f"{result.symbol} [{result.timeframe}]: {status} "
                f"(Score: {result.quality_score:.2%})"
            )
            lines.append(f"  Bars: {result.total_bars:,}")
            lines.append(
                f"  Missing: {result.missing_bars}, "
                f"OHLC Violations: {result.ohlc_violations}, "
                f"Volume Anomalies: {result.volume_anomalies}, "
                f"Price Anomalies: {result.price_anomalies}"
            )

            if result.issues:
                lines.append("  Issues:")
                for issue in result.issues[:5]:
                    lines.append(f"    - {issue}")

            lines.append("")

        valid_count = sum(1 for result in results if result.is_valid)
        lines.append("-" * 60)
        lines.append(f"SUMMARY: {valid_count}/{len(results)} datasets passed validation")
        return "\n".join(lines)

    def _normalize_timeframes(self, timeframes: Optional[Sequence[str]]) -> Optional[set[str]]:
        """Normalizes optional timeframe filters to cache suffix values."""
        if not timeframes:
            return None

        normalized = {str(timeframe).strip().lower() for timeframe in timeframes if str(timeframe).strip()}
        unsupported = sorted(tf for tf in normalized if tf not in self.SUPPORTED_TIMEFRAMES)
        if unsupported:
            supported = ", ".join(self.SUPPORTED_TIMEFRAMES)
            raise ValueError(
                f"Unsupported timeframe filter(s): {', '.join(unsupported)}. Supported: {supported}."
            )
        return normalized

    def _parse_cache_filename(self, cache_file: Path) -> Optional[Tuple[str, str]]:
        """Parses one cache filename into ``(symbol, timeframe)``."""
        parts = cache_file.stem.rsplit("_", 1)
        if len(parts) != 2:
            return None

        symbol, timeframe = parts[0].upper(), parts[1].lower()
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            return None

        return symbol, timeframe

    def _load_cache_frame(self, cache_file: Path) -> pd.DataFrame:
        """Loads one cache parquet file with a normalized datetime index."""
        df = pd.read_parquet(cache_file)
        if df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)

        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        return df.sort_index()
