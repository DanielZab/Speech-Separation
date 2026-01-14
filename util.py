from torch.nn.utils.rnn import pad_sequence
import torch


def get_PIT(out_1, out_2, ref_1, ref_2, metric):
    loss1_1 = metric(out_1, ref_1)
    loss1_2 = metric(out_2, ref_2)
    loss_1 = (loss1_1 + loss1_2) * 0.5
    loss2_1 = metric(out_2, ref_1)
    loss2_2 = metric(out_1, ref_2)
    loss_2 = (loss2_1 + loss2_2) * 0.5
    loss = loss = torch.maximum(loss_1, loss_2)
    return loss


def collate_pad(batch):
    xs, ys, zs = zip(*batch)

    xs = [x.transpose(0, 1) for x in xs]
    ys = [y.transpose(0, 1) for y in ys]
    zs = [z.transpose(0, 1) for z in zs]

    xs = pad_sequence(xs, batch_first=True)
    ys = pad_sequence(ys, batch_first=True)
    zs = pad_sequence(zs, batch_first=True)

    xs = xs.transpose(1, 2)
    ys = ys.transpose(1, 2)
    zs = zs.transpose(1, 2)

    return xs, ys, zs
