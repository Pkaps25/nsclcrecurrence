import os
import glob
import pandas as pd
import click


def get_label(path):
    """
    Determines the label based on the file path prefix.
    Returns:
        2 if the path starts with 'T',
        1 if the path starts with 'S',
        0 otherwise.
    """
    if path.startswith("T"):
        return 2
    return int(path.startswith("S"))


@click.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the .nrrd files.",
)
@click.option(
    "--output-file",
    "-o",
    required=True,
    type=click.Path(writable=True),
    help="Path to save the transformed annotations CSV file.",
)
def process_annotations(input_dir, output_file):
    """
    Processes .nrrd files and their corresponding .fcsv.gz files
    to generate a transformed annotations CSV.
    """
    cols = [
        "id",
        "x",
        "y",
        "z",
        "ow",
        "ox",
        "oy",
        "oz",
        "vis",
        "sel",
        "lock",
        "label",
        "desc",
        "associatedNodeID",
    ]
    annotations = []

    # Gather .nrrd files
    files = glob.glob(os.path.join(input_dir, "*nrrd"))

    for file_path in files:
        path = os.path.basename(file_path)
        annotation_file_path = os.path.join(input_dir, f"{path.replace('nrrd', 'fcsv')}.gz")
        if not os.path.exists(annotation_file_path):
            click.echo(f"Skipping {file_path}: Missing corresponding .fcsv.gz file")
            continue

        # Read the annotation file
        annotation_file = pd.read_csv(
            annotation_file_path, names=cols, index_col=False, comment="#"
        )

        # Extract coordinates and create annotations
        for x, y, z in annotation_file[["x", "y", "z"]].values:
            annotations.append([file_path, x, y, z, get_label(path)])

    # Save to CSV
    df = pd.DataFrame(annotations, columns=["path", "x", "y", "z", "label"])
    df.to_csv(output_file, index=False)
    click.echo(f"Annotations saved to {output_file}")


if __name__ == "__main__":
    process_annotations()
