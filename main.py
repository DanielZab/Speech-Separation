import os
from torchaudio.models import ConvTasNet, conv_tasnet_base
import torch
from functools import partial
import uuid

import tqdm
from dataloader import CustomDataset
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
from torch.utils.tensorboard import SummaryWriter
from util import collate_pad, get_PIT

TRAIN_TEST_SPLIT = 0.6
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
EPOCHS = 10
CACHE_THRESHOLD = 1e10


def empty_cache():
    if torch.cuda.memory_reserved() > CACHE_THRESHOLD:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def print_memory():
    print(torch.cuda.memory_allocated() / 1e06, "MB allocated")
    print(torch.cuda.memory_reserved() / 1e06, "MB reserved")


def main():
    torch.manual_seed(67)

    run_id = uuid.uuid4().hex
    print(f"Run ID: {run_id}")

    if not os.path.exists("runs/"):
        os.makedirs("runs/")

    writer = SummaryWriter(log_dir=f"runs/{run_id}")

    dataset = CustomDataset(
        root_dir="LibriMix/data/Libri2Mix/wav8k/max",
        sub_dirs=["train-100", "train-360"],
    )

    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset,
        [
            TRAIN_TEST_SPLIT,
            1 - TRAIN_TEST_SPLIT,
        ],
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pad,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_pad,
    )

    print("Dataset loaded.")
    print(f"Number of training samples: {len(dataset_train)}")
    print(f"Number of validation samples: {len(dataset_val)}")

    if not os.path.exists("checkpoints/"):
        os.makedirs("checkpoints/")

    if not os.path.exists(f"checkpoints/{run_id}/"):
        os.makedirs(f"checkpoints/{run_id}/")

    model = ConvTasNet(
        num_sources=2,
        enc_kernel_size=16,
        enc_num_feats=256,  # originally 512
        msk_kernel_size=3,
        msk_num_feats=64,  # originally 128
        msk_num_hidden_feats=256,  # originally 512
        msk_num_layers=8,
        msk_num_stacks=3,
        msk_activate="relu",
    )
    val_names = [
        # "pesq",
        "stoi",
    ]
    val_metrics = [
        #      partial(perceptual_evaluation_speech_quality, fs=8000, mode="nb"),
        partial(short_time_objective_intelligibility, fs=8000),
    ]

    loss_fn = scale_invariant_signal_distortion_ratio
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, maximize=True)
    # torch.cuda.memory._record_memory_history()
    best = None
    try:
        for epoch in range(EPOCHS):
            for i, batch in enumerate(
                tqdm.tqdm(train_loader, desc=f"Training epoch {epoch}")
            ):
                model.train()
                optimizer.zero_grad()
                mix, s1, s2 = batch
                mix = mix.to(DEVICE)
                s1 = s1.to(DEVICE)
                s2 = s2.to(DEVICE)

                entry = epoch * len(train_loader) + i
                est_sources = model(mix)
                shape = est_sources.shape
                loss_train = torch.mean(
                    get_PIT(
                        est_sources[:, 0, :].view(shape[0], 1, shape[2]),
                        est_sources[:, 1, :].view(shape[0], 1, shape[2]),
                        s1,
                        s2,
                        loss_fn,
                    )
                )
                writer.add_scalar(
                    "Train/Loss",
                    loss_train.detach().item(),
                    entry,
                )
                loss_train.backward()

                optimizer.step()
                empty_cache()

            vals = []
            with torch.no_grad():
                for i, batch in enumerate(
                    tqdm.tqdm(val_loader, desc=f"Validation epoch {epoch}")
                ):
                    model.eval()
                    mix, s1, s2 = batch
                    mix = mix.to(DEVICE)
                    s1 = s1.to(DEVICE)
                    s2 = s2.to(DEVICE)
                    entry = epoch * len(val_loader) + i
                    est_sources = model(mix)
                    shape = est_sources.shape
                    loss = torch.mean(
                        get_PIT(
                            est_sources[:, 0, :].view(shape[0], 1, shape[2]),
                            est_sources[:, 1, :].view(shape[0], 1, shape[2]),
                            s1,
                            s2,
                            loss_fn,
                        )
                    )

                    vals.append(loss.item())
                    writer.add_scalar(
                        "Val/Loss",
                        loss.item(),
                        entry,
                    )

                    for j, metric in enumerate(val_metrics):
                        val_metric = torch.mean(
                            get_PIT(
                                est_sources[:, 0, :].cpu().view(shape[0], 1, shape[2]),
                                est_sources[:, 1, :].cpu().view(shape[0], 1, shape[2]),
                                s1.cpu(),
                                s2.cpu(),
                                metric,
                            )
                        )

                        writer.add_scalar(
                            f"Val/{val_names[j]}_metric",
                            val_metric.item(),
                            entry,
                        )

                avg_val = sum(vals) / len(vals)
                if best is None or avg_val > best:
                    best = avg_val
                    torch.save(
                        model.state_dict(), f"checkpoints/{run_id}/best_model.pt"
                    )
                    print(f"Epoch {epoch}: New best model saved with val loss {best}")
    except Exception as e:
        print("Training crashed:", e)
    finally:
        torch.save(model.state_dict(), f"checkpoints/{run_id}/final_model.pt")
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        writer.close()


if __name__ == "__main__":
    main()
