from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm


def group_dataset_by_paper_url(df):
    """
    Group dataset by paper_url with specified aggregation rules:
    - Remove: qa_id, question, answer, passage_position
    - Pick first: paper_id, paper_title, year, venue
    - Append all (deduplicated): passage_id, passage_text, specialty
    """

    # Define aggregation functions
    agg_functions = {
        "paper_id": "first",
        "paper_title": "first",
        "year": "first",
        "venue": "first",
        "passage_id": lambda x: list(set(x)),  # Remove duplicates and convert to list
        "passage_text": lambda x: list(set(x)),  # Remove duplicates and convert to list
        "specialty": lambda x: list(set(x)),  # Remove duplicates and convert to list
    }

    # Group by paper_url and apply aggregations
    grouped_df = df.groupby("paper_url").agg(agg_functions).reset_index()

    # Optional: Sort lists for consistency (you can remove this if order doesn't matter)
    grouped_df["passage_id"] = grouped_df["passage_id"].apply(
        lambda x: sorted(x) if isinstance(x, list) else x
    )
    grouped_df["passage_text"] = grouped_df["passage_text"].apply(
        lambda x: sorted(x) if isinstance(x, list) else x
    )
    grouped_df["specialty"] = grouped_df["specialty"].apply(
        lambda x: sorted(x) if isinstance(x, list) else x
    )

    return grouped_df


def main():
    print("Loading Miriad dataset...")
    dataset = load_dataset("miriad/miriad-5.8M", split="train")
    print("Miriad dataset loaded successfully.")
    print("Converting to pandas DataFrame...")
    df = pd.DataFrame(dataset)
    print("Conversion to DataFrame completed.")
    df["passage_text"] = df.apply(
        lambda x: x["paper_title"] + "\n" + x["passage_text"], axis=1
    )
    # Get unique passage_text content and create mapping
    unique_passages = df["passage_text"].drop_duplicates()
    passage_to_id = {passage: idx for idx, passage in enumerate(unique_passages)}
    df["passage_id"] = df["passage_text"].map(passage_to_id)
    dataset.add_column("passage_id", df["passage_id"].tolist())
    print("Passage IDs added to the dataset.")
    print("Grouping dataset by paper_url...")
    df = pd.DataFrame(dataset)
    grouped_df = group_dataset_by_paper_url(df)
    print("Grouping completed.")
    print("Push processed dataset to Hugging Face Hub...")
    dataset = Dataset.from_pandas(grouped_df)
    dataset.push_to_hub("hoanganhpham/miriad-5.8M-processed")


if __name__ == "__main__":
    main()
