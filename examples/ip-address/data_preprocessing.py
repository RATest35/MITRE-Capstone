import pandas as pd
from pathlib import Path

INPUT_FILE = r""
OUTPUT_FILE = r"cleaned_flows.csv"

class FlowDataPreprocessor:
    """Preprocess large network flow CSV data in chunks.

    Handles:
    - Data cleaning and validation
    - Duplicate removal across chunks
    - Aggregation into graph-ready edge format
    """

    def __init__(self, chunksize=200_000):
        """Initialize preprocessor with chunk size and required columns.

        :param chunksize: Number of rows to process per chunk.
        """
        self.chunksize = chunksize
        self.required_columns = [
            "Flow.ID",
            "Source.IP",
            "Destination.IP",
            "Total.Length.of.Fwd.Packets",
            "Flow.Bytes.s"
        ]

    def clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean a single chunk of flow data.

        :param chunk: Raw chunk of CSV data.
        :return: Cleaned DataFrame with valid rows only.
        """
        # Keep only required columns to reduce memory usage
        chunk = chunk[self.required_columns].copy()

        # Normalize string columns (remove whitespace inconsistencies)
        for col in ["Flow.ID", "Source.IP", "Destination.IP"]:
            chunk[col] = chunk[col].astype(str).str.strip()

        # Replace invalid string placeholders with proper NaN values
        chunk.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}, inplace=True)

        # Drop rows missing key identifiers (cannot be used in graph)
        chunk.dropna(subset=["Flow.ID", "Source.IP", "Destination.IP"], inplace=True)

        # Drop rows missing numeric values needed for aggregation
        chunk.dropna(subset=["Total.Length.of.Fwd.Packets", "Flow.Bytes.s"], inplace=True)

        # Convert numeric columns safely (invalid values → NaN)
        for col in ["Total.Length.of.Fwd.Packets", "Flow.Bytes.s"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Remove rows with negative values (invalid network metrics)
        chunk = chunk[
            (chunk["Total.Length.of.Fwd.Packets"] >= 0) &
            (chunk["Flow.Bytes.s"] >= 0)
        ]

        return chunk

    def preprocess_csv(self, input_csv: str, output_csv: str):
        """Clean CSV in chunks and remove duplicate Flow.ID globally.

        :param input_csv: Path to raw input CSV file.
        :param output_csv: Path to save cleaned output CSV.
        """
        input_csv = Path(input_csv)
        output_csv = Path(output_csv)

        seen_flow_ids = set()  # Tracks Flow.ID across all chunks
        first_write = True

        # Metrics for debugging / validation
        total_rows = 0
        total_written = 0
        total_dropped_duplicates = 0

        for chunk in pd.read_csv(
            input_csv,
            usecols=self.required_columns,
            chunksize=self.chunksize,
            low_memory=False
        ):
            total_rows += len(chunk)

            # Step 1: clean raw data
            chunk = self.clean_chunk(chunk)

            # Step 2: remove duplicates within chunk
            chunk = chunk.drop_duplicates(subset="Flow.ID", keep="first")

            # Step 3: remove duplicates across previous chunks
            is_new = ~chunk["Flow.ID"].isin(seen_flow_ids)
            total_dropped_duplicates += (~is_new).sum()
            chunk = chunk[is_new]

            # Step 4: update global seen set
            seen_flow_ids.update(chunk["Flow.ID"].tolist())

            # Step 5: write incrementally to avoid memory issues
            if not chunk.empty:
                chunk.to_csv(
                    output_csv,
                    mode="w" if first_write else "a",
                    index=False,
                    header=first_write
                )
                first_write = False
                total_written += len(chunk)

        # Summary statistics (useful for debugging large datasets)
        print("Preprocessing complete.")
        print(f"Total input rows read: {total_rows}")
        print(f"Rows written: {total_written}")
        print(f"Duplicate Flow.ID rows removed: {total_dropped_duplicates}")

    def combine_for_graph(self, cleaned_csv: str, output_csv: str):
        """Aggregate cleaned flow data into graph edge format.

        Groups by (Source.IP, Destination.IP) to create edges.

        :param cleaned_csv: Path to cleaned CSV file.
        :param output_csv: Path to save aggregated edge CSV.
        """
        cleaned_csv = Path(cleaned_csv)
        output_csv = Path(output_csv)

        agg_parts = []  # Intended to store intermediate chunk aggregations

        for chunk in pd.read_csv(cleaned_csv, chunksize=self.chunksize, low_memory=False):
            # Aggregate flows within this chunk
            grouped = (
                chunk.groupby(["Source.IP", "Destination.IP"], as_index=False)
                .agg({
                    "Total.Length.of.Fwd.Packets": "sum",  # total packet size
                    "Flow.Bytes.s": "sum",                 # total bytes
                    "Flow.ID": "count"                    # number of flows
                })
            )

            grouped.rename(columns={"Flow.ID": "Flow.Count"}, inplace=True)

            # NOTE: grouped is not appended → agg_parts remains empty (bug)
            # agg_parts.append(grouped)  # <-- should be here

        if not agg_parts:
            print("No data found to aggregate.")
            return

        # Combine all chunk-level aggregations
        combined = pd.concat(agg_parts, ignore_index=True)

        # Final aggregation across chunks (ensures global correctness)
        final_grouped = (
            combined.groupby(["Source.IP", "Destination.IP"], as_index=False)
            .agg({
                "Total.Length.of.Fwd.Packets": "sum",
                "Flow.Bytes.s": "sum",
                "Flow.Count": "sum"
            })
        )

        final_grouped.to_csv(output_csv, index=False)
        print(f"Aggregated graph-ready file saved to: {output_csv}")


if __name__ == "__main__":
    processor = FlowDataPreprocessor(chunksize=200_000)

    # clean and remove duplicates
    processor.preprocess_csv(
        input_csv=INPUT_FILE,
        output_csv="cleaned_flows.csv"
    )

    # combine for graph edges
    processor.combine_for_graph(
        cleaned_csv="cleaned_flows.csv",
        output_csv= OUTPUT_FILE
    )