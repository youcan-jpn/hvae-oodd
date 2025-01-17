import argparse
import copy
import datetime
import logging
import os
import random
from uuid import uuid4
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import rich
import sklearn.metrics
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

import oodd
import oodd.models
import oodd.datasets
import oodd.variational
import oodd.losses

from oodd.utils import str2bool, get_device, log_sum_exp, set_seed, plot_gallery
from oodd.evaluators import Evaluator


LOGGER = logging.getLogger(name=__file__)


try:
    import wandb
    wandb_available = True
except ImportError:
    LOGGER.warning("Running without remote tracking!")
    wandb_available = False


parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument("--model", default="VAE", help="model type (vae | lvae | biva)")
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
parser.add_argument("--train_samples", type=int, default=1, help="samples from approximate posterior")
parser.add_argument("--test_samples", type=int, default=1, help="samples from approximate posterior")
parser.add_argument("--train_importance_weighted", type=str2bool, default=False, const=True, nargs="?", help="use iw bound")
parser.add_argument("--test_importance_weighted", type=str2bool, default=False, const=True, nargs="?", help="use iw bound")
parser.add_argument("--warmup_epochs", type=int, default=0, help="epochs to warm up the KL term.")
parser.add_argument("--free_nats_epochs", type=int, default=0, help="epochs to warm up the KL term.")
parser.add_argument("--free_nats", type=float, default=0, help="nats considered free in the KL term")
parser.add_argument("--n_eval_samples", type=int, default=32, help="samples from prior for quality inspection")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed")
parser.add_argument("--test_every", type=int, default=1, help="epochs between evaluations")
parser.add_argument("--save_dir", type=str, default="./models", help="directory for saving models")
parser.add_argument("--use_wandb", type=str2bool, default=True, help="use wandb tracking")
parser.add_argument("--name", type=str, default="hvae-oodd", help="wandb tracking name")
parser.add_argument("--checkpoint_path", type=str, default="", help="path to checkpoint")
parser.add_argument("--exp_id", type=str, default="", help="unique id for the experiment")

parser = oodd.datasets.DataModule.get_argparser(parents=[parser])

args, unknown_args = parser.parse_known_args()

args.start_time = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
args.train_sample_reduction = log_sum_exp if args.train_importance_weighted else torch.mean
args.test_sample_reduction = log_sum_exp if args.test_importance_weighted else torch.mean
args.use_wandb = wandb_available and args.use_wandb


set_seed(args.seed)


LoadingRet = namedtuple(
    "LoadingRet",
    [
        "epoch",
        "model",
        "datamodule",
        "optimizer",
    ]
)


def ddp_setup(rank: int, world_size: int):
    """
    Parameters
    ----------
    rank : int
        Unique identifier of each process
    world_size : int
        Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_cleanup():
    dist.destroy_process_group()


def save_checkpoint(save_dir, *, epoch, optimizer):
    OTHERS_CHECKPOINT_NAME = "others.pt"
    cp = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        "cuda_random": torch.cuda.get_rng_state(),
        "cuda_random_all": torch.cuda.get_rng_state_all(),
    }
    save_path = os.path.join(save_dir, OTHERS_CHECKPOINT_NAME)
    torch.save(cp, save_path)


def save_checkpoint_with_epoch(save_dir, *, epoch, optimizer):
    OTHERS_CHECKPOINT_NAME = f"others_{epoch}.pt"
    cp = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        "cuda_random": torch.cuda.get_rng_state(),
        "cuda_random_all": torch.cuda.get_rng_state_all(),
    }
    save_path = os.path.join(save_dir, OTHERS_CHECKPOINT_NAME)
    torch.save(cp, save_path)


def start_from_scratch(rank: int) -> LoadingRet:
    print("starting from scratch ...")
    # Data
    datamodule = oodd.datasets.DataModule(
        batch_size=args.batch_size,
        test_batch_size=250,
        data_workers=args.data_workers,
        train_datasets=args.train_datasets,
        val_datasets=args.val_datasets,
        test_datasets=args.test_datasets,
    )
    in_shape = datamodule.train_dataset.datasets[0].size[0]

    args.save_dir = os.path.join(
        args.save_dir,
        list(datamodule.train_datasets.keys())[0] + "-" + args.exp_id
    )

    os.makedirs(args.save_dir, exist_ok=True)

    if rank == 0:
        datamodule.save(args.save_dir)

    fh = logging.FileHandler(os.path.join(args.save_dir, "dvae.log"))
    fh.setLevel(logging.INFO)
    LOGGER.addHandler(fh)

    # Model
    model = getattr(oodd.models.dvae, args.model)
    model_argparser = model.get_argparser()
    model_args, unknown_model_args = model_argparser.parse_known_args()
    model_args.input_shape = in_shape
    model = model(**vars(model_args)).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return LoadingRet(
        model=model,
        epoch=0,
        datamodule=datamodule,
        optimizer=optimizer,
    )


def start_from_checkpoint(rank: int) -> LoadingRet:
    print("starting from checkpoint ...")
    checkpoint = oodd.models.Checkpoint(path=args.checkpoint_path)
    others_path = os.path.join(args.checkpoint_path, "others.pt")
    checkpoint.load_DDP(rank=rank, others_path=others_path)
    model = checkpoint.model
    datamodule = checkpoint.datamodule
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.load_state_dict(checkpoint.others["optimizer"])
    epoch = checkpoint.others["epoch"]  # TODO

    args.exp_id = _get_exp_name_from_checkpoint_dir(args.checkpoint_path)
    args.save_dir = os.path.join(
        args.save_dir,
        list(datamodule.train_datasets.keys())[0] + "-" + args.exp_id
    )
    print(f"{args.save_dir=}")

    fh = logging.FileHandler(os.path.join(args.save_dir, "dvae.log"))
    fh.setLevel(logging.INFO)
    LOGGER.addHandler(fh)

    return LoadingRet(
        model=model,
        epoch=epoch,
        datamodule=datamodule,
        optimizer=optimizer,
    )


def _get_exp_name_from_checkpoint_dir(checkpoint_dir: str):
    separated = checkpoint_dir.split("-")
    if len(separated) == 2:
        return separated[1]
    else:
        raise ValueError("Unknown type of checkpoint folder name")


def train(epoch, rank, model, deterministic_warmup, free_nats_cooldown, datamodule, criterion, optimizer):
    model.train()
    evaluator = Evaluator(primary_metric="log p(x)", logger=LOGGER, use_wandb=args.use_wandb)

    beta = next(deterministic_warmup)
    free_nats = next(free_nats_cooldown)

    if rank == 0:
        iterator = tqdm(enumerate(datamodule.train_loader), smoothing=0.9, total=len(datamodule.train_loader), leave=False)
    else:
        iterator = enumerate(datamodule.train_loader)

    for _, (x, _) in iterator:
        x = x.to(rank)

        likelihood_data, stage_datas = model(x, n_posterior_samples=args.train_samples)
        kl_divergences = [
            stage_data.loss.kl_elementwise for stage_data in stage_datas if stage_data.loss.kl_elementwise is not None
        ]
        loss, elbo, likelihood, kl_divergences = criterion(
            likelihood_data.likelihood,
            kl_divergences,
            samples=args.train_samples,
            free_nats=free_nats,
            beta=beta,
            sample_reduction=args.train_sample_reduction,
            batch_reduction=None,
        )

        l = loss.mean()
        l.backward()

        optimizer.step()
        optimizer.zero_grad()

        evaluator.update("Train", "elbo", {"log p(x)": elbo})
        evaluator.update("Train", "likelihoods", {"loss": -loss, "log p(x)": elbo, "log p(x|z)": likelihood})
        klds = {
            f"KL z{i+1}": kl
            for i, kl in enumerate([sd.loss.kl_samplewise for sd in stage_datas if sd.loss.kl_samplewise is not None])
        }
        klds["KL(q(z|x), p(z))"] = kl_divergences
        evaluator.update("Train", "divergences", klds)

    dist.barrier()
    evaluator.update(
        "Train", "hyperparameters", {"free_nats": [free_nats], "beta": [beta], "learning_rate": [args.learning_rate]}
    )
    evaluator.report(epoch * len(datamodule.train_loader))
    evaluator.log(epoch)


@torch.no_grad()
def test(epoch, rank, model, dataloader, evaluator, criterion, in_shape, dataset_name="test", max_test_examples=float("inf")):
    LOGGER.info(f"Testing: {dataset_name}")
    model.eval()

    x, _ = next(iter(dataloader))
    x = x.to(rank)
    n = min(x.size(0), 8)
    likelihood_data, stage_datas = model(x, n_posterior_samples=args.test_samples)
    p_x_mean = likelihood_data.mean[: args.batch_size].view(args.batch_size, *in_shape)  # Reshape zeroth "sample"
    p_x_samples = likelihood_data.samples[: args.batch_size].view(args.batch_size, *in_shape)  # Reshape zeroth "sample"
    comparison = torch.cat([x[:n], p_x_mean[:n], p_x_samples[:n]])
    comparison = comparison.permute(0, 2, 3, 1)  # [B, H, W, C]
    fig, ax = plot_gallery(comparison.cpu().numpy(), ncols=n)
    fig.savefig(os.path.join(args.save_dir, f"reconstructions_{dataset_name}_{epoch:03}"))
    plt.close()

    decode_from_p_combinations = [[True] * n_p + [False] * (model.module.n_latents - n_p) for n_p in range(model.module.n_latents)]
    for decode_from_p in tqdm(decode_from_p_combinations, leave=False):
        n_skipped_latents = sum(decode_from_p)

        if max_test_examples != float("inf"):
            iterator = tqdm(
                zip(range(max_test_examples // dataloader.batch_size), dataloader),
                smoothing=0.9,
                total=max_test_examples // dataloader.batch_size,
                leave=False,
            )
        else:
            iterator = tqdm(enumerate(dataloader), smoothing=0.9, total=len(dataloader), leave=False)

        for _, (x, _) in iterator:
            x = x.to(rank)

            likelihood_data, stage_datas = model(
                x, n_posterior_samples=args.test_samples, decode_from_p=decode_from_p, use_mode=decode_from_p
            )
            kl_divergences = [
                stage_data.loss.kl_elementwise
                for stage_data in stage_datas
                if stage_data.loss.kl_elementwise is not None
            ]
            loss, elbo, likelihood, kl_divergences = criterion(
                likelihood_data.likelihood,
                kl_divergences,
                samples=args.test_samples,
                free_nats=0,
                beta=1,
                sample_reduction=args.test_sample_reduction,
                batch_reduction=None,
            )

            if n_skipped_latents == 0:  # Regular ELBO
                evaluator.update(dataset_name, "elbo", {"log p(x)": elbo})
                evaluator.update(
                    dataset_name, "likelihoods", {"loss": -loss, "log p(x)": elbo, "log p(x|z)": likelihood}
                )
                klds = {
                    f"KL z{i+1}": kl
                    for i, kl in enumerate(
                        [sd.loss.kl_samplewise for sd in stage_datas if sd.loss.kl_samplewise is not None]
                    )
                }
                klds["KL(q(z|x), p(z))"] = kl_divergences
                evaluator.update(dataset_name, "divergences", klds)

            evaluator.update(dataset_name, f"skip-elbo", {f"{n_skipped_latents} log p(x)": elbo})
            evaluator.update(dataset_name, f"skip-elbo-{dataset_name}", {f"{n_skipped_latents} log p(x)": elbo})
            evaluator.update(
                dataset_name,
                f"skip-likelihoods-{dataset_name}",
                {
                    f"{n_skipped_latents} loss": -loss,
                    f"{n_skipped_latents} log p(x)": elbo,
                    f"{n_skipped_latents} log p(x|z)": likelihood,
                },
            )
            klds = {
                f"{n_skipped_latents} KL z{i+1}": kl
                for i, kl in enumerate(
                    [sd.loss.kl_samplewise for sd in stage_datas if sd.loss.kl_samplewise is not None]
                )
            }
            klds[f"{n_skipped_latents} KL(q(z|x), p(z))"] = kl_divergences
            evaluator.update(dataset_name, f"skip-divergences-{dataset_name}", klds)


def collapse_multiclass_to_binary(y_true, zero_label=None):
    # Force the class index in zero_label to be zero and the others to collapse to 1
    zero_label_indices = y_true == zero_label
    y_true[zero_label_indices] = 0
    y_true[~zero_label_indices] = 1
    return y_true


def compute_roc_auc(y_true=None, y_score=None, zero_label=None):
    """Compares class zero_label to all other classes in y_true"""
    y_true = collapse_multiclass_to_binary(y_true, zero_label)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score, average="macro")
    return roc_auc, fpr, tpr, thresholds


def compute_pr_auc(y_true=None, y_score=None, zero_label=None):
    """Compares class zero_label to all other classes in y_true"""
    y_true = collapse_multiclass_to_binary(y_true, zero_label)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    pr_auc = sklearn.metrics.average_precision_score(y_true, y_score, average="macro")
    return pr_auc, precision, recall, thresholds


def compute_roc_pr_metrics(y_true, y_score, classes, reference_class):
    """Compute the ROC and PR metrics from a primary dataset class to a number of other dataset classes"""
    roc_results = {}
    pr_results = {}
    for class_label in sorted(set(y_true) - set([reference_class])):
        idx = np.logical_or(y_true == reference_class, y_true == class_label)  # Compare primary to the other dataset

        roc_auc, fpr, tpr, thresholds = compute_roc_auc(
            y_true=y_true[idx], y_score=y_score[idx], zero_label=reference_class
        )

        pr_auc, precision, recall, thresholds = compute_pr_auc(
            y_true=y_true[idx], y_score=y_score[idx], zero_label=reference_class
        )

        idx_where_tpr_is_eighty = np.where((tpr - 0.8 >= 0))[0][0]
        fpr80 = fpr[idx_where_tpr_is_eighty]

        ood_target = [source for source, label in classes.items() if label == class_label][0]
        roc_results[ood_target] = dict(roc_auc=roc_auc, fpr=fpr, tpr=tpr, fpr80=fpr80, thresholds=thresholds)
        pr_results[ood_target] = dict(pr_auc=pr_auc, precision=precision, recall=recall, thresholds=thresholds)

    return roc_results, pr_results


def subsample_labels_and_scores(y_true, y_score, n_examples):
    """Subsample y_true and y_score to have n_examples while maintaining their relative ordering"""
    assert len(y_true) == len(y_score) >= n_examples, f"Got {len(y_true)=}, {len(y_score)=}, {n_examples=}"
    indices = [np.random.choice(np.where(y_true == i)[0], n_examples, replace=False) for i in set(y_true)]
    y_true = np.concatenate([y_true[idx] for idx in indices])
    y_score = np.concatenate([y_score[idx] for idx in indices])
    return y_true, y_score


def main(rank: int, world_size: int, exp_id: str):
    if args.use_wandb and rank == 0:
        wandb.init(args.name)
    elif args.use_wandb:
        args.use_wandb = False  # turn off if the rank != 0
    print(f"Running DDP on rank {rank}")
    ddp_setup(rank, world_size)
    if args.exp_id == "":
        args.exp_id = exp_id

    if args.checkpoint_path == "":
        settings = start_from_scratch(rank=rank)
    else:
        settings = start_from_checkpoint(rank=rank)
    model = settings.model
    datamodule = settings.datamodule
    optimizer = settings.optimizer
    start_epoch = settings.epoch

    deterministic_warmup = oodd.variational.DeterministicWarmup(
        n=args.warmup_epochs,
        start_epoch=start_epoch,
    )
    free_nats_cooldown = oodd.variational.FreeNatsCooldown(
        constant_epochs=args.free_nats_epochs // 2,
        cooldown_epochs=args.free_nats_epochs // 2,
        start_val=args.free_nats,
        end_val=0,
        start_epoch=start_epoch,
    )

    criterion = oodd.losses.ELBO()

    in_shape = datamodule.train_dataset.datasets[0].size[0]
    p_z_samples = model.module.prior.sample(torch.Size([args.n_eval_samples])).to(rank)
    sample_latents = [None] * (model.module.n_latents - 1) + [p_z_samples]

    # Logging
    LOGGER.info("Experiment config:")
    LOGGER.info(args)
    rich.print(vars(args))
    LOGGER.info("%s", deterministic_warmup)
    LOGGER.info("%s", free_nats_cooldown)
    LOGGER.info("DataModule:\n%s", datamodule)
    LOGGER.info("Model:\n%s", model)

    # Run
    test_elbos = [-np.inf]
    test_evaluator = Evaluator(primary_source=datamodule.primary_val_name, primary_metric="log p(x)", logger=LOGGER, use_wandb=args.use_wandb)

    LOGGER.info("Running training...")
    for current_epoch in range(start_epoch + 1, args.epochs + 1):
        datamodule.train_loader.sampler.set_epoch(current_epoch)

        train(
            epoch=current_epoch,
            rank=rank,
            model=model,
            deterministic_warmup=deterministic_warmup,
            free_nats_cooldown=free_nats_cooldown,
            datamodule=datamodule,
            criterion=criterion,
            optimizer=optimizer,
        )

        dist.barrier()

        if current_epoch % args.test_every == 0 and rank == 0:
            # Sample
            with torch.no_grad():
                likelihood_data, stage_datas = model.module.sample_from_prior(
                    n_prior_samples=args.n_eval_samples, forced_latent=sample_latents
                )
                p_x_samples = likelihood_data.samples.view(args.n_eval_samples, *in_shape)
                p_x_mean = likelihood_data.mean.view(args.n_eval_samples, *in_shape)
                comparison = torch.cat([p_x_samples, p_x_mean])
                comparison = comparison.permute(0, 2, 3, 1)  # [B, H, W, C]
                fig, ax = plot_gallery(comparison.cpu().numpy(), ncols=args.n_eval_samples // 4)
                fig.savefig(os.path.join(args.save_dir, f"samples_{current_epoch:03}"))
                plt.close()

            # Test
            for name, dataloader in datamodule.val_loaders.items():
                test(
                    epoch=current_epoch,
                    rank=rank,
                    model=model,
                    dataloader=dataloader,
                    evaluator=test_evaluator,
                    criterion=criterion,
                    in_shape=in_shape,
                    dataset_name=name,
                    max_test_examples=10000,
                )

            # Save
            test_elbo = test_evaluator.get_primary_metric().mean().cpu().numpy()
            if current_epoch % 50 == 0:
                model.module.save_with_epoch(args.save_dir, rank=rank, epoch=current_epoch)
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"ddp_model_state_dict_{current_epoch}.pt"))
                save_checkpoint_with_epoch(
                    args.save_dir, epoch=current_epoch, optimizer=optimizer
                )
            if np.max(test_elbos) < test_elbo:
                test_evaluator.save(args.save_dir)
                model.module.save(args.save_dir, rank=rank)
                torch.save(model.state_dict(), os.path.join(args.save_dir, "ddp_model_state_dict.pt"))
                save_checkpoint(
                    args.save_dir, epoch=current_epoch, optimizer=optimizer
                )
                LOGGER.info("Saved model!")
            test_elbos.append(test_elbo)

            # Compute LLR
            for source in test_evaluator.sources:
                for k in range(1, model.module.n_latents):
                    log_p_a = test_evaluator.metrics[source][f"skip-elbo"][f"0 log p(x)"]
                    log_p_b = test_evaluator.metrics[source][f"skip-elbo"][f"{k} log p(x)"]
                    llr = log_p_a - log_p_b
                    test_evaluator.update(source, series="LLR", metrics={f"LLR>{k}": llr})

            # Compute AUROC score for L>k and LLR>k metrics
            reference_dataset = datamodule.primary_val_name
            max_examples = min(
                [len(d) for d in datamodule.val_datasets.values()]
            ) // world_size  # Maximum number of examples to use for equal sized sets

            # L >k
            for n_skipped_latents in range(model.module.n_latents):
                y_true, y_score, classes = test_evaluator.get_classes_and_scores_per_source(
                    f"skip-elbo", f"{n_skipped_latents} log p(x)"
                )
                y_true, y_score = subsample_labels_and_scores(y_true, y_score, max_examples)
                roc, pr = compute_roc_pr_metrics(
                    y_true, -y_score, classes, classes[reference_dataset]
                )  # Negation since higher score means more OOD
                for ood_target, value_dict in roc.items():
                    test_evaluator.update(
                        source=reference_dataset,
                        series=f"ROC AUC L>k",
                        metrics={f"ROC AUC L>{n_skipped_latents} {ood_target}": [value_dict["roc_auc"]]},
                    )
                    test_evaluator.update(
                        source=reference_dataset,
                        series=f"ROC AUC L>{n_skipped_latents}",
                        metrics={f"ROC AUC L>{n_skipped_latents} {ood_target}": [value_dict["roc_auc"]]},
                    )

            # LLR >0 >k
            for n_skipped_latents in range(1, model.module.n_latents):
                y_true, y_score, classes = test_evaluator.get_classes_and_scores_per_source(
                    "LLR", f"LLR>{n_skipped_latents}"
                )
                y_true, y_score = subsample_labels_and_scores(y_true, y_score, max_examples)
                roc, pr = compute_roc_pr_metrics(y_true, y_score, classes, classes[reference_dataset])
                for ood_target, value_dict in roc.items():
                    test_evaluator.update(
                        source=reference_dataset,
                        series=f"ROC AUC LLR>k",
                        metrics={f"ROC AUC LLR>{n_skipped_latents} {ood_target}": [value_dict["roc_auc"]]},
                    )
                    test_evaluator.update(
                        source=reference_dataset,
                        series=f"ROC AUC LLR>{n_skipped_latents}",
                        metrics={f"ROC AUC LLR>{n_skipped_latents} {ood_target}": [value_dict["roc_auc"]]},
                    )

            # Report
            test_evaluator.report(current_epoch * len(datamodule.train_loader))
            test_evaluator.log(current_epoch)
            test_evaluator.reset()
    ddp_cleanup()


def run_main(main_fn, world_size: int, exp_id: str):
    mp.spawn(main_fn, args=(world_size, exp_id), nprocs=world_size, join=True)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    default_exp_id = uuid4().hex
    run_main(main_fn=main, world_size=world_size, exp_id=default_exp_id)
