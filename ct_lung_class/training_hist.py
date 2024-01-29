import argparse
import copy
import datetime
import os
import sys

import numpy as np

#from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from dsets_hist import NoduleDataset
from util.logconf import logging
#from model import NoduleModel
#from res_model import NoduleModel
import monai

log = logging.getLogger(__name__)
filename='log_hist_DN_drop5_aug_vals4_boxv64_SGD.log'
file_handler = logging.FileHandler(filename)
OUTPUT_PATH = "/home/kaplinsp/ct_lung_class/ct_lung_class/models/"
log.addHandler(file_handler)
log.setLevel(logging.INFO)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

class NoduleTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=8,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=50,
            type=int,
        )
        parser.add_argument('--balanced',
            help="Balance the training data to half positive, half negative.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=True,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=True,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=True,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=True,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=True,
        )

        #parser.add_argument('--tb-prefix',
         #   default='p2ch12',
          #  help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        #)
        #parser.add_argument('comment',
         #   help="Comment suffix for Tensorboard run.",
          #  nargs='?',
           # default='dlwpt',
        #)

        parser.add_argument('--finetune',
            help="Start finetuning from this model.",
            default='',
        )

        parser.add_argument('--finetune-depth',
            help="Number of blocks (counted from head) to include in finetuning",
            type=int,
            default=1,
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        #self.trn_writer = None
        #self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        #    self.augmentation_dict['offset'] = 0.1
         #   self.augmentation_dict['offset'] = 0.1
          #  self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # if not self.use_cuda:
        #     raise ValueError("CUDA not enabled, don't train")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()


    def initModel(self):
        #model = monai.networks.nets.densenet.densenet121(pretrained=True,spatial_dims=3,in_channels=1,out_channels=2)
        #model = NoduleModel()
        model = monai.networks.nets.densenet.densenet121(pretrained=False,dropout_prob=0.5,spatial_dims=3,in_channels=1,out_channels=2)
        #model = monai.networks.nets.densenet.densenet121(pretrained=False,dropout_prob=0.2,spatial_dims=3,in_channels=1,out_channels=2)
 
        if self.cli_args.finetune:
            d=torch.load(self.cli_args.finetune, map_location='cpu')
            model_blocks= [
                n for n, subm in model.named_children()
                if len(list(subm.parameters()))>0
            ]
            finetune_blocks=model_blocks[-self.cli_args.finetune_depth:]
            #log.info(f"finetuning from {self.cli_args.finetune}, blocks {' ', join(finetune_blocks)}")
            model.load_state_dict(
                {
                    k: v for k,v in d['model_state'].items()
                    if k.split('.')[0] not in model_blocks[-1]
                },
                strict=False,
            )
            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count()>1:
                model=nn.DataParallel(model)
            model=model.to(self.device)
        return model


    def initOptimizer(self):
        #lr=0.003 if self.cli_args.finetune else 0.001
        lr=0.001
        #return SGD(self.model.parameters(),lr=lr, weight_decay=1e-4)       
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        #return Adam(self.model.parameters(),lr)

    def initTrainDl(self):
        train_ds = NoduleDataset(
            val_stride=4,
            isValSet_bool=False,
            ratio_int=0,
            augmentation_dict=self.augmentation_dict,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = NoduleDataset(
           # val_stride=10,
            val_stride=4,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

#    def initTensorboardWriters(self):
 #       if self.trn_writer is None:
  #          log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

   #         self.trn_writer = SummaryWriter(
    #            log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
     #       self.val_writer = SummaryWriter(
      #          log_dir=log_dir + '-val_cls-' + self.cli_args.comment)


    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        bestm1 = -1
        bestm2 = -1
        bestm3 = -1

        best_paths = [None, None, None]

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            m1, m2,m3= self.logMetrics(epoch_ndx, 'val', valMetrics_t)

            if m1>bestm1: # best val F1
                bestm1=m1
                best_paths[0] = (copy.deepcopy(self.model.state_dict()), epoch_ndx)
            if m2>bestm2: # best val accuracy
                bestm2=m2
                best_paths[1] = (copy.deepcopy(self.model.state_dict()), epoch_ndx)
            if m3>bestm3: # best val accuracy
                bestm3=m3
                best_paths[2] = (copy.deepcopy(self.model.state_dict()), epoch_ndx)


        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        
        for i, (model_state, epoch_ndx) in enumerate(best_paths):
            torch.save(
                model_state,
                os.path.join(OUTPUT_PATH, f"bestm{i+1}_model_transform_{filename}_e{epoch_ndx}.pth")
            )

      #  if hasattr(self, 'trn_writer'):
       #     self.trn_writer.close()
        #    self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            torch.autograd.set_detect_anomaly(True)
            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')


    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                )
        return valMetrics_g.to('cpu')



    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        #label_t[np.where(label_t==0)]=-1

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        #logits_g, probability_g = self.model(input_g)
        logits_g = self.model(input_g)
        foo=np.exp(logits_g.detach().cpu().numpy())/np.exp(logits_g.detach().cpu().numpy()).sum(1)[:,None]
        probability_g=torch.tensor(foo)

        #class_weights=[1,2]
        #class_weights = torch.FloatTensor(class_weights)
        #loss_func = nn.CrossEntropyLoss(reduction='none',weight=class_weights)

        ###### HINGE LOSS
        #loss_func=nn.HingeEmbeddingLoss()
        #loss_g=loss_func(logits_g,label_g)
 
        #########CROSS ENTROPY LOSS
        loss_func = nn.CrossEntropyLoss()
        loss_g = loss_func(logits_g,label_g[:,1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:,1]
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:,1]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        return loss_g.mean()


    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float32(truePos_count + falsePos_count)
        recall    = metrics_dict['pr/recall'] = \
            truePos_count / np.float32(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        log.info(
            ("E{} {:8} {loss/all:.8f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.8f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.8f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )
       
        return metrics_dict['pr/f1_score'],metrics_dict['correct/all'],metrics_dict['loss/all']
        #writer = getattr(self, mode_str + '_writer')

        #for key, value in metrics_dict.items():
         #   writer.add_scalar(key, value, self.totalTrainingSamples_count)

        #writer.add_pr_curve(
         #   'pr',
          #  metrics_t[METRICS_LABEL_NDX],
          #  metrics_t[METRICS_PRED_NDX],
           # self.totalTrainingSamples_count,
        #)



    # def logModelMetrics(self, model):
    #     writer = getattr(self, 'trn_writer')
    #
    #     model = getattr(model, 'module', model)
    #
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             min_data = float(param.data.min())
    #             max_data = float(param.data.max())
    #             max_extent = max(abs(min_data), abs(max_data))
    #
    #             # bins = [x/50*max_extent for x in range(-50, 51)]
    #
    #             try:
    #                 writer.add_histogram(
    #                     name.rsplit('.', 1)[-1] + '/' + name,
    #                     param.data.cpu().numpy(),
    #                     # metrics_a[METRICS_PRED_NDX, negHist_mask],
    #                 log.error([min_data, max_data])
    #                 raise


if __name__ == '__main__':
    NoduleTrainingApp().main()

