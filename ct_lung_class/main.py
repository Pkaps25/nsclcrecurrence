from util.logconf import logging
from util.util import importstr

log = logging.getLogger("nb")

# module avail
# module load cuda/11.3
# module load cudnn/8.1.0-cuda11.2
# bsub -Is -q gpuqueue -n 6 -gpu "num=1" -R V100 -R "rusage[mem=6] span[hosts=1]" -W 5:00 python run_stuff.py


def run(app, *argv):
    argv = list(argv)
    log.info(f"Running: {app}({argv}).main()")

    app_cls = importstr(*app.rsplit(".", 1))
    app_cls(argv).main()

    log.info(f"Finished: {app}.({argv}).main()")


training_epochs = 1
batch_size = 32
run(
    "train.NoduleTrainingApp",
    f"--epochs={training_epochs}",
    f"--batch-size={batch_size}",
    "--num-workers=24",
    "--augment-flip",
    "--augment-offset",
    "--augment-rotate",
    "--augment-noise",
)
