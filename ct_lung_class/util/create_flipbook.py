import os
import random
import click
import matplotlib.pyplot as plt
from datasets import getCtRawNodule, getNoduleInfoList
from pypdf import PdfWriter


@click.command()
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    help="Directory to save the individual PDFs.",
)
@click.option(
    "--merged-pdf",
    "-m",
    required=True,
    type=click.Path(writable=True),
    help="Path to save the merged PDF.",
)
@click.option(
    "--datasets",
    "-d",
    multiple=True,
    type=str,
    required=True,
    help="List of dataset names to process nodules from.",
)
def process_nodules(output_dir, merged_pdf, datasets):
    """
    Processes nodules, generates individual PDFs with their visualizations, and merges them into a single PDF.
    """
    nodules = getNoduleInfoList(datasets)

    # Shuffle nodules
    random.shuffle(nodules)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = []

    for i, nod in enumerate(nodules):
        try:
            # Generate the nodule image
            image = getCtRawNodule(
                nod.file_path, nod.image_type, nod.center_lps, True, 10, [128] * 3, [50, 50, 50]
            )
            center_slice = image[0][64]

            # Save individual PDF
            pdf_path = os.path.join(output_dir, f"nodule{i}.pdf")
            plt.imshow(center_slice, cmap="gray")
            plt.title(
                f"Class label: {'SCLC' if nod.is_nodule else 'NSCLC'}\nCase ID: {nod.file_path}"
            )
            plt.savefig(pdf_path)
            plt.close()
            pdf_files.append(pdf_path)

        except Exception as e:
            click.echo(f"Error processing nodule {i}: {e}")
            continue

    # Merge PDFs
    merger = PdfWriter()
    for pdf in pdf_files:
        merger.append(pdf)

    merger.write(merged_pdf)
    merger.close()

    click.echo(f"Merged PDF saved to {merged_pdf}")


if __name__ == "__main__":
    process_nodules()
