import pandas as pd
from pathlib import Path

INPUT_FILE = r""
OUTPUT_FILE = r"cleaned_flows.csv"

class FlowDataPreprocessor:
    def __init__(self, chunksize=200_000):
        self.chunksize = chunksize
        self.required_columns = ["Flow.ID", "Source.IP", "Destination.IP", "Total.Length.of.Fwd.Packets",
                                 "Flow.Bytes.s"]

    def clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Clean one chunk of the dataset.
        """
        # keep only need columns
        chunk = chunk[self.required_columns].copy()

        # standardize whitespace in strings
        for col in ["Flow.ID", "Source.IP", "Destination.IP"]:
            chunk[col] = chunk[col].astype(str).str.strip()

        #  empty strings/invalid string nulls with replaced with NaN
        chunk.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}, inplace=True)

        # drop rows missing identifiers
        chunk.dropna(subset=["Flow.ID", "Source.IP", "Destination.IP"], inplace=True)

        # drop rows where numeric fields are missing
        chunk.dropna(subset=["Total.Length.of.Fwd.Packets", "Flow.Bytes.s"], inplace=True)

        # columns expected to hold number values are changed to appropriate data types
        for col in ["Total.Length.of.Fwd.Packets", "Flow.Bytes.s"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Remove negative values if they exist
        chunk = chunk[(chunk["Total.Length.of.Fwd.Packets"] >= 0) & (chunk["Flow.Bytes.s"] >= 0)]

        return chunk

    def preprocess_csv(self, input_csv: str, output_csv: str):
        """
        Clean the dataset in chunks and remove duplicate Flow.ID rows globally.
        Assumes duplicate Flow.ID rows should only appear once in the output.
        """
        input_csv = Path(input_csv)
        output_csv = Path(output_csv)

        seen_flow_ids = set()
        first_write = True
        total_rows = 0
        total_written = 0
        total_dropped_duplicates = 0

        for chunk in pd.read_csv(input_csv, usecols=self.required_columns, chunksize=self.chunksize, low_memory=False):
            total_rows += len(chunk)

            # clean chunk
            chunk = self.clean_chunk(chunk)

            # remove duplicates inside this chunk first
            chunk = chunk.drop_duplicates(subset="Flow.ID", keep="first")

            # remove duplicates across previous chunks
            is_new = ~chunk["Flow.ID"].isin(seen_flow_ids)
            total_dropped_duplicates += (~is_new).sum()
            chunk = chunk[is_new]

            # update seen set
            seen_flow_ids.update(chunk["Flow.ID"].tolist())

            # write to output
            if not chunk.empty:
                chunk.to_csv(
                    output_csv,
                    mode="w" if first_write else "a",
                    index=False,
                    header=first_write
                )
                first_write = False
                total_written += len(chunk)

        print("Preprocessing complete.")
        print(f"Total input rows read: {total_rows}")
        print(f"Rows written: {total_written}")
        print(f"Duplicate Flow.ID rows removed: {total_dropped_duplicates}")

    def combine_for_graph(self, cleaned_csv: str, output_csv: str):
        """
        Combine cleaned flow-level data into edge-level data for graph modeling.
        Groups by Source.IP and Destination.IP.
        """
        cleaned_csv = Path(cleaned_csv)
        output_csv = Path(output_csv)

        agg_parts = []

        for chunk in pd.read_csv(cleaned_csv, chunksize=self.chunksize, low_memory=False):
            grouped = (
                chunk.groupby(["Source.IP", "Destination.IP"], as_index=False)
                .agg({
                    "Total.Length.of.Fwd.Packets": "sum",
                    "Flow.Bytes.s": "sum",
                    "Flow.ID": "count"
                })
            )

            grouped.rename(columns={"Flow.ID": "Flow.Count"}, inplace=True)

        if not agg_parts:
            print("No data found to aggregate.")
            return

        combined = pd.concat(agg_parts, ignore_index=True)

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