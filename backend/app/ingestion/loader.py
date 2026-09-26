from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class DatasetLoadError(Exception):
    """
    Raised when a dataset cannot be loaded.
    """

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _make_bad_line_handler(skipped_lines: List[Dict[str, Any]]):
    """
    Pandas callback used for malformed CSV rows.
    """

    def handle_bad_line(bad_line):
        skipped_lines.append(
            {
                "line": len(skipped_lines) + 1,
                "values": bad_line,
            }
        )

        return None

    return handle_bad_line


def _load_csv(
    file_path: str,
    encoding: str,
    skipped_lines: List[Dict[str, Any]],
    delimiter: str | None = None,
) -> pd.DataFrame:

    bad_line_handler = _make_bad_line_handler(skipped_lines)

    try:
        if delimiter is None:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines=bad_line_handler,
            )
        else:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                engine="python",
                on_bad_lines=bad_line_handler,
            )

        return df

    except pd.errors.EmptyDataError as exc:
        raise DatasetLoadError(
            "EMPTY_FILE",
            "The CSV file is empty and contains no data."
        ) from exc

    except pd.errors.ParserError as exc:
        raise DatasetLoadError(
            "PARSER_ERROR",
            "The CSV file could not be parsed correctly."
        ) from exc

    except UnicodeDecodeError as exc:
        raise DatasetLoadError(
            "ENCODING_ERROR",
            f"The file could not be decoded using {encoding}."
        ) from exc

    except Exception as exc:
        raise DatasetLoadError(
            "CSV_LOAD_ERROR",
            f"Unable to read the CSV file: {str(exc)}"
        ) from exc


def _load_csv_with_fallbacks(file_path: str):
    """
    Try multiple encodings and delimiter detection.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252",
    ]

    delimiters = [
        ",",
        ";",
        "\t",
        "|",
    ]

    last_error = None

    for encoding in encodings:

        # First try Pandas automatic delimiter detection.
        skipped_lines: List[Dict[str, Any]] = []

        try:
            df = _load_csv(
                file_path,
                encoding,
                skipped_lines,
                delimiter=None,
            )

            if df.shape[1] > 1:
                return df, {
                    "encoding": encoding,
                    "delimiter": "auto-detected",
                    "skipped_rows": skipped_lines,
                }

        except DatasetLoadError as exc:
            last_error = exc

        # If automatic detection produced one column,
        # explicitly test common delimiters.
        for delimiter in delimiters:

            skipped_lines = []

            try:
                df = _load_csv(
                    file_path,
                    encoding,
                    skipped_lines,
                    delimiter=delimiter,
                )

                if df.shape[1] > 1:
                    return df, {
                        "encoding": encoding,
                        "delimiter": delimiter,
                        "skipped_rows": skipped_lines,
                    }

            except DatasetLoadError as exc:
                last_error = exc

    if last_error:
        raise last_error

    raise DatasetLoadError(
        "DELIMITER_ERROR",
        "Unable to detect a valid delimiter for the CSV file."
    )


def _load_excel(file_path: str):
    try:
        df = pd.read_excel(
            file_path,
            engine="openpyxl"
        )

        return df, {
            "encoding": None,
            "delimiter": None,
            "skipped_rows": [],
        }

    except ValueError as exc:
        raise DatasetLoadError(
            "EXCEL_FORMAT_ERROR",
            "The Excel file could not be read. Make sure it is a valid .xlsx file."
        ) from exc

    except Exception as exc:
        raise DatasetLoadError(
            "EXCEL_LOAD_ERROR",
            f"Unable to read the Excel file: {str(exc)}"
        ) from exc


def load_dataset(file_path: str):
    """
    Load CSV or XLSX dataset.

    Returns:
        dataframe
        loading metadata
    """

    extension = Path(file_path).suffix.lower()

    if extension == ".csv":
        return _load_csv_with_fallbacks(file_path)

    if extension == ".xlsx":
        return _load_excel(file_path)

    raise DatasetLoadError(
        "UNSUPPORTED_FORMAT",
        "Unsupported file format. Only CSV and XLSX files are supported."
    )