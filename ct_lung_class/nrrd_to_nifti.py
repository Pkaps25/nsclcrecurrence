import click
import pandas as pd
import os
import SimpleITK as sitk


@
@click.command()
@click.option('--annotations', '-a', required=True, type=click.Path(exists=True), help="Path to the annotations CSV file.")
@click.option('--output-dir', '-o', required=True, type=click.Path(file_okay=False, writable=True), help="Directory to save the converted NIfTI files.")
def run(annotations, output_dir):
    """
    Converts images listed in an annotations CSV to NIfTI format and saves them to the specified directory.
    """
    annots = pd.read_csv(annotations)

    for path in annots['path'].values:
        fname = os.path.basename(path).split(".")[0]
        out_path = os.path.join(output_dir, f"{fname}_0000.nii.gz")
        if os.path.exists(out_path):
            click.echo(f"Skipping {out_path}")
            continue
        img = sitk.ReadImage(path)
        sitk.WriteImage(img, out_path)
        click.echo(f"Processed and saved: {out_path}")


if __name__ == "__main__":
    run()
