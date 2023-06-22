import copy
import functools
import os
import random
import time
from types import SimpleNamespace

import blobfile as bf
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from data_utils.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler
from diffusion.resample import create_named_schedule_sampler
from eval import eval_humanml
from eval.eval_mixamo import mixamo_evaluate
from sample.generate import main as generate
from utils import dist_util

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0




class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.crop_ratio = args.crop_ratio
        self.sin_path = args.sin_path

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.arch = args.arch
        self.lr_method = args.lr_method
        self.lr_step = args.lr_step
        self.lr_gamma = args.lr_gamma

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.lr_method == 'StepLR':
            assert self.lr_step is not None
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.lr_step, gamma=self.lr_gamma)
        elif self.lr_method == 'ExponentialLR':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=self.lr_gamma)
        else:
            assert self.lr_method is None, f'lr scheduling {self.lr_method} is not supported.'

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper = None
        if args.dataset == 'humanml' and args.eval_during_training:
            self.num_samples_limit = 100
            self.replication_times = 3
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def adjust_learning_rate(self, optimizer, step, args):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if hasattr(self, 'lr_scheduler'):
            return
        if step < args.warmup_steps:
            lr = args.lr * step / args.warmup_steps
        else:
            lr = args.lr
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)


    def run_loop(self, motion):

        if self.args.dataset == 'mixamo':
            n_jointsfeatures, n_frames = motion.shape
        else:
            n_joints, n_features, n_frames = motion.shape

        batch = motion.repeat((self.batch_size, 1, 1, 1))
        cond = {'y': {'mask': None}}

        start_time_measure = time.time()
        time_measure = []
        for self.step in range(self.num_steps-self.resume_step):
            self.run_step(batch, cond)
            if self.total_step() % self.log_interval == 0:
                for k, v in logger.get_current().name2val.items():
                    if k == 'loss':
                        print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                    if k in ['step', 'samples'] or '_q' in k:
                        continue
                    else:
                        self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')
                if self.total_step() > 0:
                    end_time_measure = time.time()
                    elapsed = end_time_measure - start_time_measure
                    time_measure.append(elapsed)
                    print(f'Time of last {self.log_interval} iterations: {int(elapsed)} seconds.')
                    start_time_measure = time.time()
                self.train_platform.report_scalar(name='Learning Rate', value=self.opt.param_groups[0]['lr'], iteration=self.total_step(), group_name='LR')

            if self.total_step() % self.save_interval == 0 and self.total_step() != 0 or self.total_step() == self.num_steps - 1:
                self.save()
                self.model.eval()
                self.evaluate()
                self.generate_during_training()
                self.model.train()

                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                    return
            self.adjust_learning_rate(self.opt, self.step, self.args)

        if len(time_measure) > 0:
            mean_times = sum(time_measure) / len(time_measure)
            print(f'Average time for {self.log_interval} iterations: {mean_times} seconds.')


    def print_changed_lr(self, lr_saved):
        lr_cur = self.opt.param_groups[0]['lr']
        if lr_saved is not None:
            if abs(lr_saved - lr_cur) > 0.05 * lr_saved:
                print(f'step {self.total_step()}: lr_saved updated to ', lr_cur)
                lr_saved =lr_cur
        else:
            lr_saved = lr_cur
            print(f'step {self.total_step()}: lr_saved = ', lr_cur)
        return lr_saved

    def total_step(self):
        return self.step + self.resume_step


    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        eval_dict = {}
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 20 min]')
            eval_dict = eval_humanml.evaluate(self.args, self.model, self.diffusion, self.eval_wrapper,
                                                     self.num_samples_limit, self.replication_times)
            for win_size, metrics in eval_dict.items():
                for m, vals in metrics.items():
                    self.train_platform.report_scalar(name=m, value=vals['mean'],
                                                      iteration=self.total_step(),
                                                      group_name=f'Eval_window{win_size}')
        elif self.dataset == 'mixamo':
            eval_dict = mixamo_evaluate(self.args, self.model, self.diffusion)
            print(eval_dict)
            for m, vals in eval_dict.items():
                self.train_platform.report_scalar(name=m, value=vals['mean'], iteration=self.total_step(),
                                                  group_name=f'Eval')
        log_file = os.path.join(self.save_dir, f'eval_{self.total_step():09d}.log')
        with open(log_file, 'w') as fw:
            fw.write(str(eval_dict))
        np.save(log_file.replace('.log', '.npy'), eval_dict)
        print(eval_dict)
        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval - start_eval) / 60}min')

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # random resize
            if self.arch == 'unet' or self.arch == 'qna':
                curr_w = round(micro.shape[3] * random.uniform(0.75, 1.25))
                curr_w = 4 * (curr_w // 4)
                micro = F.interpolate(micro, (micro.shape[2], curr_w), mode="bicubic")

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset if self.data is not None else None
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def log_step(self):
        logger.logkv("step", self.total_step() + self.resume_step)
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{self.total_step():09d}.pt"

    def generate_during_training(self):
        if not self.args.gen_during_training:
            return
        gen_args = copy.deepcopy(self.args)
        gen_args.model_path = os.path.join(self.save_dir, self.ckpt_file_name())
        gen_args.output_dir = os.path.join(self.save_dir, f'{self.ckpt_file_name()}.samples')
        gen_args.num_samples = self.args.gen_num_samples
        gen_args.motion_length = None
        all_sample_save_path = generate(gen_args)
        self.train_platform.report_media(title='Motion', series='Predicted Motion', iteration=self.total_step(),
                                         local_path=all_sample_save_path)

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{self.total_step():09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
