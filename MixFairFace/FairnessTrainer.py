import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loops import FitLoop
from pytorch_lightning.loops.dataloader import EvaluationLoop
from pytorch_lightning.loops.dataloader.evaluation_loop import _select_data_fetcher_type
from pytorch_lightning.loops.epoch import TrainingEpochLoop, EvaluationEpochLoop
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from itertools import chain
from typing import Any, IO, Iterable, List, Optional, Sequence, Type, Union
from . import Tools, Datasets, Verify


def PrepareDataset(config):
    dataset_func_train = Tools.rgetattr(Datasets, config['dataset_args']['train']['dataset_type'])
    dataset_func_val = Tools.rgetattr(Datasets, config['dataset_args']['val']['dataset_type'])
    dataset_func_val_inst = Tools.rgetattr(Datasets, config['dataset_args']['val-instance']['dataset_type'])
    train_data = dataset_func_train(**config['dataset_args']['train'])
    val_data = dataset_func_val(**config['dataset_args']['val'])
    val_inst_data =  dataset_func_val_inst(**config['dataset_args']['val-instance'])

    return train_data, val_data, val_inst_data      

def ScriptStart(args, config, litmodule):
    num_nodes = config['exp_args'].get('num_nodes', 1)
    devices = config['exp_args'].get('devices', None)
    if args.mode == 'train':
        logger = TensorBoardLogger(config['exp_args']['exp_path'])
        trainer = FairnessTrainer(
            accelerator="gpu", 
            strategy='ddp', 
            enable_progress_bar=False,
            max_epochs=config['exp_args']['epoch'],
            num_sanity_val_steps=0,
            logger=logger,
            num_nodes=num_nodes,
            devices=devices
        )
        trainer.fit(model=litmodule)
    elif args.mode == 'val':
        trainer = FairnessTrainer(
            accelerator="gpu", 
            strategy='ddp',
            enable_progress_bar=False,
            max_epochs=config['exp_args']['epoch'],
            num_sanity_val_steps=0,
            logger=False,
            num_nodes=num_nodes,
            devices=devices
        )
        cvl = CustomValLoop()
        trainer.validate_loop = cvl
        trainer.validate(model=litmodule)
        if litmodule.global_rank == 0:
            print(litmodule.ValResults())
    elif args.mode == 'val-inst':
        trainer = FairnessTrainer(
            accelerator="gpu", 
            strategy='ddp', 
            enable_progress_bar=False,
            max_epochs=config['exp_args']['epoch'],
            num_sanity_val_steps=0,
            logger=False,
            num_nodes=num_nodes,
            devices=devices
        )
        cvl = CustomValInstLoop()
        trainer.validate_loop = cvl
        trainer.validate(model=litmodule)
        if litmodule.global_rank == 0:
            litmodule.PrintValInstResults(litmodule.ValInstResults())
            feats = litmodule.ValInstFeatures()
            litmodule.SaveFeatures(feats, 999)
    elif args.mode == 'val-inst-run':
        feats_path = args.feats
        print ('Load features %s'%feats_path)
        feats = torch.FloatTensor(np.load(feats_path))
        results = litmodule.val_inst_analyze(feats)
        litmodule.PrintValInstResults(results)
        np.save('results.npy', results)


class CustomEvaluationLoop(EvaluationLoop):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)
    
    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        super().on_advance_start(*args, **kwargs)
        
        self.a = CustomValLoop()
        self.a.trainer = self.trainer
        self.b = CustomValInstLoop()
        self.b.trainer = self.trainer
    
    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.lightning_module.SetValMode()
        self.trainer.reset_val_dataloader()
        self.a.run()
        self.trainer.lightning_module.SetValInstMode()
        self.trainer.reset_val_dataloader()
        self.b.run()

class CustomValLoop(EvaluationLoop):
    def _reload_evaluation_dataloaders(self) -> None:
        """Reloads dataloaders if necessary."""
        self.trainer.lightning_module.SetValMode()
        if self.trainer.testing:
            self.trainer.reset_test_dataloader()
        elif self.trainer.val_dataloaders is None or self.trainer._data_connector._should_reload_val_dl:
            self.trainer.reset_val_dataloader()


class CustomValInstLoop(EvaluationLoop):
    def _reload_evaluation_dataloaders(self) -> None:
        """Reloads dataloaders if necessary."""
        self.trainer.lightning_module.SetValInstMode()
        if self.trainer.testing:
            self.trainer.reset_test_dataloader()
        elif self.trainer.val_dataloaders is None or self.trainer._data_connector._should_reload_val_dl:
            self.trainer.reset_val_dataloader()

    def _on_evaluation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_epoch_start`` and ``on_{validation/test}_epoch_start`` hooks."""
        self.trainer._logger_connector.on_epoch_start()
        self.trainer._call_callback_hooks("on_epoch_start", *args, **kwargs)
        self.trainer._call_lightning_module_hook("on_epoch_start", *args, **kwargs)

        if self.trainer.testing:
            self.trainer._call_callback_hooks("on_test_epoch_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_test_epoch_start", *args, **kwargs)
        else:
            self.trainer._call_callback_hooks("on_validation_epoch_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_validation_inst_epoch_start", *args, **kwargs)

    def _evaluation_epoch_end(self, outputs: List[EPOCH_OUTPUT]) -> None:
        """Runs ``{validation/test}_epoch_end``"""
        self.trainer._logger_connector._evaluation_epoch_end()

        # with a single dataloader don't pass a 2D list
        output_or_outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]] = (
            outputs[0] if len(outputs) > 0 and self.num_dataloaders == 1 else outputs
        )

        # call the model epoch end
        if self.trainer.testing:
            self.trainer._call_lightning_module_hook("test_epoch_end", output_or_outputs)
        else:
            self.trainer._call_lightning_module_hook("validation_inst_epoch_end", output_or_outputs)


class FairnessCallback(pl.callbacks.Callback):
    def __init__(self, tqdm_total, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_total = tqdm_total

    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.global_rank == 0:
            count = int(np.ceil(len(pl_module.train_dataloader_obj) / self.tqdm_total))
            self.myprogress = Tools.MyTqdm(range(count))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if pl_module.global_rank == 0: next(self.myprogress)

    def on_validation_start(self, trainer, pl_module):
        if pl_module.global_rank == 0:
            tmp = len(pl_module.val_dataloader_obj) if pl_module.val_mode == 0 else len(pl_module.val_inst_dataloader_obj)
            count = int(np.ceil(tmp / self.tqdm_total))
            self.myprogress = Tools.MyTqdm(range(count))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.global_rank == 0: next(self.myprogress)


class FairnessTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #cvl = CustomEvaluationLoop()
        cvl = CustomValLoop()
        epoch_loop = TrainingEpochLoop()
        epoch_loop.val_loop = cvl
        self.fit_loop = FitLoop(max_epochs=self.max_epochs)
        self.fit_loop.epoch_loop = epoch_loop

        tqdm_total = self.num_nodes * self.num_devices
        self.callbacks.append(FairnessCallback(tqdm_total=tqdm_total))

class FairnessLightningModule(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.val_mode = 0
        self.val_bag = dict()
        self.val_results = None
        self.val_inst_bag = dict()
        self.val_inst_results = None

    def SetDataset(self, train_data, val_data, val_inst_data):
        self.train_data = train_data
        self.val_data = val_data
        self.val_inst_data = val_inst_data

        self.train_dataloader_obj = train_data.CreateLoader()
        self.val_dataloader_obj = val_data.CreateLoader()
        self.val_inst_dataloader_obj = val_inst_data.CreateLoader()
    
    def train_dataloader(self):
        return self.train_dataloader_obj

    def val_dataloader(self):
        if self.val_mode == 0: return self.val_dataloader_obj
        else: return self.val_inst_dataloader_obj
    
    def SetValMode(self):
        self.val_mode = 0
    
    def SetValInstMode(self):
        self.val_mode = 1
    
    def ValResults(self):
        return self.val_results
    
    def ValInstResults(self):
        return self.val_inst_results
    
    def ValInstFeatures(self):
        return self.val_inst_bag['feats']

    def WriteValResults(self, results):
        writer = self.logger.experiment
        for key, val in results.items(): 
            if key == 'group':
                for race, race_val in val.items():
                    writer.add_scalar('Val/%s/%s'%(key, race), race_val, self.current_epoch)
            else:
                writer.add_scalar('Val/%s'%key, val, self.current_epoch)

    def WriteValInstResults(self, results):
        writer = self.logger.experiment
        param_a = {
            'title': 'Inter-identity Similarity',
            'xlabel': 'ID',
            'ylabel': 'Similarity'
        }
        param_b = {
            'title': 'Intra-identity Similarity',
            'xlabel': 'ID',
            'ylabel': 'Similarity'
        }
        inter_fig = Tools.PlotRaceDistribution(results['inter_sim_race'], self.val_inst_data.GetRaceList(), param=param_a)
        intra_fig = Tools.PlotRaceDistribution(results['intra_sim_race'], self.val_inst_data.GetRaceList(), param=param_b)
        writer.add_image('Inst-Intra Similarity', intra_fig, self.current_epoch, dataformats='HWC')
        writer.add_image('Inst-Inter Similarity', inter_fig, self.current_epoch, dataformats='HWC')

        writer.add_scalar('Inst-eval/thr', results['thr'], self.current_epoch)
        writer.add_scalar('Inst-eval/overall-tpr', results['tpr'], self.current_epoch)
        writer.add_scalar('Inst-eval/overall-fpr', results['fpr'], self.current_epoch)

        writer.add_scalar('Inst-eval/instance-avg-tpr', results['tpr_inst'].mean(), self.current_epoch)
        writer.add_scalar('Inst-eval/instance-avg-fpr', results['fpr_inst'].mean(), self.current_epoch)
        writer.add_scalar('Inst-eval/instance-std-tpr', results['tpr_inst'].std(), self.current_epoch)
        writer.add_scalar('Inst-eval/instance-std-fpr', results['fpr_inst'].std(), self.current_epoch)

        for i, race in enumerate(self.val_inst_data.GetRaceList()):
            writer.add_scalar('Inst-eval/race-avg-tpr-(%s)'%race, results['tpr_race'][i].mean(), self.current_epoch)
            writer.add_scalar('Inst-eval/race-std-tpr-(%s)'%race, results['tpr_race'][i].std(), self.current_epoch)
            writer.add_scalar('Inst-eval/race-avg-fpr-(%s)'%race, results['fpr_race'][i].mean(), self.current_epoch)
            writer.add_scalar('Inst-eval/race-std-fpr-(%s)'%race, results['fpr_race'][i].std(), self.current_epoch)

        for i, race in enumerate(self.val_inst_data.GetRaceList()):
            writer.add_scalar('Inst-eval/intra-sim-avg-(%s)'%race, results['intra_sim_race'][i].mean(), self.current_epoch)
            writer.add_scalar('Inst-eval/intra-sim-std-(%s)'%race, results['intra_sim_race'][i].std(), self.current_epoch)
            writer.add_scalar('Inst-eval/inter-sim-avg-(%s)'%race, results['inter_sim_race'][i].mean(), self.current_epoch)
            writer.add_scalar('Inst-eval/inter-sim-std-(%s)'%race, results['inter_sim_race'][i].std(), self.current_epoch)
    
    def PrintValResults(self, results):
        if self.global_rank == 0: print (results)

    def PrintValInstResults(self, results):
        if self.global_rank == 0 and results is not None:
            print ('Best thr: %f'%results['thr'])
            print ('Overall TPR/FPR: %f/%e' %(results['tpr'], results['fpr']))
            print ('Instance AVG TPR/FPR: %f/%e'%(results['tpr_inst'].mean(), results['fpr_inst'].mean()))
            print ('Instance STD TPR/FPR: %f/%e'%(results['tpr_inst'].std(), results['fpr_inst'].std()))

            for i, race in enumerate(self.val_inst_data.GetRaceList()):
                print ('%s AVG TPR/FPR: %f/%e'%(race, results['tpr_race'][i].mean(), results['fpr_race'][i].mean()))
                print ('%s STD TPR/FPR: %f/%e'%(race, results['tpr_race'][i].std(), results['fpr_race'][i].std()))

            for i, race in enumerate(self.val_inst_data.GetRaceList()):
                print ('Intra/Inter-similarity AVG (%s): %f %f'%(race, results['intra_sim_race'][i].mean(), results['inter_sim_race'][i].mean()))
                print ('Intra/Inter-similarity STD (%s): %f %f'%(race, results['intra_sim_race'][i].std(), results['inter_sim_race'][i].std()))

    def val_inst_analyze(self, feats):
        dataset = self.val_inst_data
        analyzer = Verify.Analysis.RFWAnalyzer(
            feats,
            dataset.GetAllID(),
            dataset.GetAllAttribute(),
            dataset.GetTFMatrix(),
            dataset.GetRaceList(),
        )
        inter_sim_race, _ = analyzer.InterIdentitySimilarity()
        intra_sim_race, _ = analyzer.IntraIdentitySimilarity()

        thr_opt = Tools.ThresholdOptimizer(feats, dataset.GetTFMatrix())
        tpr, fpr, thr, converge = thr_opt.Start(selected_fpr=1e-5)

        tpr_inst, fpr_inst, tpr_race, fpr_race = thr_opt.CalculateInstance_TPR_FPR(thr, dataset.GetAllID(), dataset.GetAllAttribute())

        results = {
            'intra_sim_race': intra_sim_race,
            'inter_sim_race': inter_sim_race,
            'tpr': tpr,
            'fpr': fpr,
            'tpr_inst': tpr_inst,
            'fpr_inst': fpr_inst,
            'tpr_race': tpr_race,
            'fpr_race': fpr_race,
            'thr': thr
        }

        return results
    
    def CreateDummyInstResults(self):
        results = {
            'intra_sim_race': [np.ones(5) for i in range(len(self.val_inst_data.GetRaceList()))],
            'inter_sim_race': [np.ones(5) for i in range(len(self.val_inst_data.GetRaceList()))],
            'tpr': 0,
            'fpr': 0,
            'tpr_inst': np.ones(5),
            'fpr_inst': np.ones(5),
            'tpr_race': [np.ones(5) for i in range(len(self.val_inst_data.GetRaceList()))],
            'fpr_race': [np.ones(5) for i in range(len(self.val_inst_data.GetRaceList()))],
            'thr': 0
        }

        return results
    
    def SaveFeatures(self, all_feats, epoch):
        assert 'feature_path' in self.config['exp_args']
        name = '%s/RFW_feats_%.5d.npy'%(self.config['exp_args']['feature_path'], epoch)
        os.system('mkdir -p %s'%self.config['exp_args']['feature_path'])
        feats = all_feats.numpy()
        np.save(name, feats)
        print ('Save features to %s'%name)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.config['learning_args']['optimizer_args']['type'])(
            self.model.parameters(),
            **self.config['learning_args']['optimizer_args']['args']
        )
        scheduler = None
        if self.config['learning_args']['scheduler_args']:
            scheduler = getattr(torch.optim.lr_scheduler, self.config['learning_args']['scheduler_args']['type'])(
                optim, **self.config['learning_args']['scheduler_args']['args']
            )
        print (optim, scheduler) 
        if scheduler is None: return [optim]
        else: return [optim], [scheduler]


    def model_forward(self, img, label):
        feat = self.extract_feature(img)
        pred = self.model.product(feat, label)

        return pred, feat

    def extract_feature(self, img):
        feat = self.model.encoder(img)['l4']
        feat = self.model.mid(feat.flatten(1))

        return feat
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.global_rank == 0:
            print ('Epoch %d/%d'%(self.current_epoch, self.config['exp_args']['epoch']-1))

    def training_step(self, batch, batch_idx):
        img_x = batch['rgb']
        class_x = batch['label']
        attribute_x = batch['attribute']
        
        pred = self.model_forward(img_x, class_x)[0]
        loss = nn.CrossEntropyLoss()(pred, class_x)

        out = {
                'loss': loss,
            }
        self.log('entropy-loss', loss, on_step=True)
        return out
    
    def on_train_epoch_end(self):
        if self.global_rank == 0:
            #self.SaveFeatures(self.ValInstFeatures(), self.current_epoch)
            print (self.ValResults())
            #self.PrintValInstResults(self.ValInstResults())
            self.WriteValResults(self.ValResults())
            #self.WriteValInstResults(self.ValInstResults())
            self.model.Save(self.current_epoch, accuracy=self.ValResults()['best_accuracy'], replace=False)

    def on_validation_epoch_start(self):
        if self.global_rank == 0:
            total = len(self.val_data)
            self.val_bag['x'] = torch.zeros(total, self.config['network_args']['head_args']['in_features'])
            self.val_bag['y'] = torch.zeros(total, self.config['network_args']['head_args']['in_features'])
            self.val_bag['label'] = torch.zeros(total, dtype=torch.bool)
            self.val_bag['attribute'] = torch.zeros(total, dtype=torch.long)

    def validation_step(self, batch, batch_idx):
        if self.val_mode == 0:
            img_x = batch['x']
            img_y = batch['y']
            label = batch['label']
            attribute = batch['attribute']
            idx = batch['idx']
            img_cat = torch.cat([img_x, img_y], dim=0)
            with torch.no_grad():
                feat1 = self.extract_feature(img_cat)
                feat2 = self.extract_feature(torch.flip(img_cat, [3]))
                feat = feat1 + feat2
            feat_x = feat[:img_cat.shape[0]//2, ...]
            feat_y = feat[img_cat.shape[0]//2:, ...]

            idx = self.all_gather(idx).flatten()
            feat_x = self.all_gather(feat_x).flatten(0, 1)
            feat_y = self.all_gather(feat_y).flatten(0, 1)
            label = self.all_gather(label).flatten()
            attribute = self.all_gather(attribute).flatten()

            if self.global_rank == 0:
                self.val_bag['x'][idx] = feat_x.cpu()
                self.val_bag['y'][idx] = feat_y.cpu()
                self.val_bag['label'][idx] = label.cpu()
                self.val_bag['attribute'][idx] = attribute.cpu()

        else:
            self.validation_inst_step(batch, batch_idx)

    def validation_epoch_end(self, val_outs):
        if self.global_rank == 0:
            x = self.val_bag['x']
            y = self.val_bag['y']
            issame = self.val_bag['label'].bool()
            attribute = self.val_bag['attribute'].long()
            #print (attribute.min(), attribute.max())

            accuracy_dict = dict()
            for i, race in enumerate(self.val_data.race_lst):
                mask = (attribute == i)
                feat_x = x[mask, :]
                feat_y = y[mask, :]
                s = issame[mask]
                tpr, fpr, accuracy, val, val_std, far = Verify.RFW.evaluate(feat_x, feat_y, s)
                accuracy_dict[race] = np.mean(accuracy)
            best_accuracy = np.mean([val*100 for key, val in accuracy_dict.items()])
            std = np.std([val*100 for key, val in accuracy_dict.items()], ddof=1)
            results = {
                'best_accuracy': best_accuracy,
                'std': std,
                'group': accuracy_dict
            }
            self.val_results = results
    
    def on_validation_inst_epoch_start(self):
        if self.global_rank == 0:
            total = len(self.val_inst_data)
            self.val_inst_bag['offset'] = 0
            self.val_inst_bag['feats'] = torch.zeros(total, self.config['network_args']['head_args']['in_features'])
    
    def validation_inst_step(self, batch, batch_idx):
        img = batch['rgb']
        idx = batch['idx']
        with torch.no_grad():
            feat1 = self.extract_feature(img)
            feat2 = self.extract_feature(torch.flip(img, [3]))
            feat = (feat1 + feat2)

        feat = self.all_gather(feat).flatten(0, 1)
        idx = self.all_gather(idx).flatten()

        if self.global_rank == 0:
            count = feat.shape[0]
            self.val_inst_bag['feats'][idx, ...] = feat.cpu()
            self.val_inst_bag['offset'] += count
    
    def validation_inst_epoch_end(self, val_inst_outs):
        if self.global_rank == 0:
            torch.set_num_threads(torch.multiprocessing.cpu_count())
            start = time.time()
            results = self.val_inst_analyze(self.val_inst_bag['feats'].cpu())
            end = time.time()
            print ('InstEval take %d seconds.'%(int(end-start)))
            self.val_inst_results = results
