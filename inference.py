import os
import torch
from dataloader import CustomDataset
from torch.utils.data import DataLoader
from util import collate_pad, get_PIT
from functools import partial
from torchaudio.models import ConvTasNet
import torchaudio
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

    mix, s1, s2 = next(iter(test_loader))

    if not os.path.exists("test_outputs/"):
        os.makedirs("test_outputs/")

    with torch.no_grad():
        est_sources = model(mix)
        torchaudio.save(
            f"test_outputs/{run_id}_s1.wav",
            est_sources[:, 0, :].cpu(),
            sample_rate=8000,
        )
        torchaudio.save(
            f"test_outputs/{run_id}_s2.wav",
            est_sources[:, 1, :].cpu(),
            sample_rate=8000,
        )


if __name__ == "__main__":
    main()
