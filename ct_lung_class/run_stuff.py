from util.logconf import logging
from util.util import importstr

log = logging.getLogger("nb")

# module avail
# module load cuda/11.3
# module load cudnn/8.1.0-cuda11.2
# bsub -Is -q gpuqueue -n 6 -gpu "num=1" -R V100 -R "rusage[mem=6] span[hosts=1]" -W 5:00 python run_stuff.py


def run(app, *argv):
    argv = list(argv)
    argv.insert(0, "--num-workers=4")
    log.info("Running: {}({!r}).main()".format(app, argv))

    app_cls = importstr(*app.rsplit(".", 1))
    app_cls(argv).main()

    log.info("Finished: {}.({!r}).main()".format(app, argv))


training_epochs = 1000
batch_size = 1
# run('training_transform_combo.NoduleTrainingApp', f'--epochs={training_epochs}','--augmented','fully-augmented')
# run('training_atrisk.NoduleTrainingApp', f'--epochs={training_epochs}','--augmented','fully-augmented')
# run('training_transform.NoduleTrainingApp', f'--epochs={training_epochs}','--augmented','fully-augmented')
run(
    "training_hist.NoduleTrainingApp",
    f"--epochs={training_epochs}",
    f"--batch-size={batch_size}",
    "--num-workers=2",
    "--augment-flip",
    "--augment-offset",
    "--augment-rotate",
    # "--augment-noise",
)
# run('training_transform.NoduleTrainingApp', f'--epochs={training_epochs}')

# run('p2ch12.training.LunaTrainingApp', f'--epochs={training_epochs}', '--balanced', '--augmented', 'fully-augmented')
