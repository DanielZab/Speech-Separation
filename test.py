import torch
from dataloader import CustomDataset
from torch.utils.data import DataLoader
from util import collate_pad, get_PIT
from functools import partial
from torchaudio.models import ConvTasNet
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio


def main():
    dataset = CustomDataset(
        root_dir="LibriMix/data/Libri2Mix/wav8k/max",
        sub_dirs=["test"],
    )

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_pad,
    )

    run_id = "b094256619dd424d8b47f03dc704123b"
    model_path = f"checkpoints/{run_id}/best_model.pt"

    val_metrics = [
        # partial(perceptual_evaluation_speech_quality, fs=8000, mode="nb"),
        partial(short_time_objective_intelligibility, fs=8000),
        scale_invariant_signal_distortion_ratio,
    ]

    val_names = [
        # "pesq",
        "stoi",
        "sdr",
    ]

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
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    model.to("cpu")

    metric_results = [[] for _ in val_metrics]
    for i, (mix, s1, s2) in enumerate(test_loader):
        if i % 10 == 9:
            break

        with torch.no_grad():
            est_sources = model(mix)
            shape = est_sources.shape

            for j, metric in enumerate(val_metrics):
                val_metric = torch.mean(
                    get_PIT(
                        est_sources[:, 0, :].view(shape[0], 1, shape[2]),
                        est_sources[:, 1, :].view(shape[0], 1, shape[2]),
                        s1,
                        s2,
                        metric,
                    )
                )

                metric_results[j].append(val_metric.item())

    for j, metric in enumerate(val_metrics):
        avg_metric = sum(metric_results[j]) / len(metric_results[j])
        print(f"{val_names[j]} average over test set: {avg_metric:.4f}")


if __name__ == "__main__":
    main()
