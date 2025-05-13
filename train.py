import os

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from AdamW import AdamWCustom
from dataset import ImageDataset
from utils import accuracy, make_directory, AverageMeter, ProgressMeter
import config
import model

# Force CPU device
device = torch.device('cpu')

class DummyOptimizer:
    def zero_grad(self): pass
    def step(self): pass


def main():
    start_epoch = 0

    train_dataloader, valid_dataloader = load_dataset()
    model_net, ema_model = build_model()
    criterion = define_loss()
    optimizer = DummyOptimizer()
    scheduler = None

    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    for epoch in range(start_epoch, config.epochs):
        train(model_net, ema_model, train_dataloader, criterion, optimizer, epoch, writer)
        acc1 = validate(ema_model, valid_dataloader, epoch, writer, "Valid")
        print("\n")


def load_dataset():
    train_dataset = ImageDataset(
        config.train_image_dir,
        config.train_annotation_path,
        config.image_size,
        "Train"
    )
    valid_dataset = ImageDataset(
        config.valid_image_dir,
        config.valid_annotation_path,
        config.image_size,
        "Valid"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=True
    )

    return train_dataloader, valid_dataloader


def build_model():
    net = model.__dict__[config.model_arch_name](
        num_classes=config.model_num_classes,
        aux_logits=True,
        transform_input=True
    )
    net = net.to(device=device)

    ema_fn = lambda avg_p, p, n: (1 - config.model_ema_decay) * avg_p + config.model_ema_decay * p
    ema_net = AveragedModel(net, avg_fn=ema_fn)
    return net, ema_net


def define_loss():
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)
    return loss_fn


def train(net, ema_net, loader, criterion, optimizer, epoch, writer):
    net.train()
    total_batches = len(loader)
    progress = ProgressMeter(
        total_batches,
        [
            AverageMeter("Loss", ":6.6f"),
            AverageMeter("Acc@1", ":6.2f"),
            AverageMeter("Acc@5", ":6.2f")
        ],
        prefix=f"Epoch: [{epoch + 1}]"
    )

    for idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = sum(
            w * criterion(o, target)
            for w, o in zip(
                [config.loss_aux3_weights, config.loss_aux2_weights, config.loss_aux1_weights],
                outputs
            )
        )
        loss.backward()
        optimizer.step()

        ema_net.update_parameters(net)

        top1, top5 = accuracy(outputs[0], target, topk=(1, 5))
        progress.meters[0].update(loss.item(), images.size(0))
        progress.meters[1].update(top1[0].item(), images.size(0))
        progress.meters[2].update(top5[0].item(), images.size(0))

        if idx % config.train_print_frequency == 0:
            writer.add_scalar("Train/Loss", loss.item(), idx + epoch * total_batches)
            progress.display(idx + 1)

    progress.display_summary('train')


def validate(net, loader, epoch, writer, mode):
    net.eval()
    total_batches = len(loader)
    progress = ProgressMeter(
        total_batches,
        [
            AverageMeter("Loss", ":6.6f"),
            AverageMeter("Acc@1", ":6.2f"),
            AverageMeter("Acc@5", ":6.2f")
        ],
        prefix=f"{mode}: "
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            target = batch["target"].to(device)

            outputs = net(images)
            loss = criterion(outputs, target)

            top1, top5 = accuracy(outputs, target, topk=(1, 5))
            progress.meters[0].update(loss.item(), images.size(0))
            progress.meters[1].update(top1[0].item(), images.size(0))
            progress.meters[2].update(top5[0].item(), images.size(0))

            if idx % config.valid_print_frequency == 0:
                progress.display(idx + 1)

    progress.display_summary('valid')

    writer.add_scalar(f"{mode}/Loss", progress.meters[0].avg, epoch + 1)
    writer.add_scalar(f"{mode}/Acc@1", progress.meters[1].avg, epoch + 1)
    writer.add_scalar(f"{mode}/Acc@5", progress.meters[2].avg, epoch + 1)

    return progress.meters[1].avg


if __name__ == "__main__":
    main()
