import argparse
import os

import paddle

from binary_cross_entropy_loss import BCELoss
from dataset import Dataset
from hed_model import HED
from hook import ComposeCallback, LogPrinter, Checkpointer
from transforms import Normalize, Resize, RandomDistort, RandomHorizontalFlip, RandomVerticalFlip


def train(model, dataloader, optimizer, loss_fn, hooks=None,
          start_epoch=0, epochs=20):
    status = {}
    status.update({
        'epoch_id': start_epoch,
        'step_id': 0,
        'steps_per_epoch': len(dataloader)
    })

    avg_loss = 0.0

    model.train()

    for epoch_id in range(start_epoch, epochs):
        status['epoch_id'] = epoch_id
        hooks.on_epoch_begin(status)

        for batch_id, (images, labels) in enumerate(dataloader):
            # images = data[0]
            # labels = data[1]

            status['step_id'] = batch_id
            hooks.on_step_begin(status)

            predictions = model(images)

            loss_list = [loss_fn(prediction, labels) for prediction in predictions]
            losses = sum(loss_list)

            losses.backward()
            optimizer.step()
            model.clear_gradients()

            lr = optimizer.get_lr()
            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_scheduler = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_scheduler = optimizer._learning_rate
            if isinstance(lr_scheduler, paddle.optimizer.lr.LRScheduler):
                lr_scheduler.step()

            avg_loss = losses.numpy().tolist()[0]

            status['loss'] = avg_loss
            status['learning_rate'] = lr

            hooks.on_step_end(status)

        hooks.on_epoch_end(status)


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of training
    parser.add_argument(
        '--epochs',
        dest='epochs',
        help='total epochs for training',
        type=int,
        default=20)

    parser.add_argument(
        '--pretrained_model',
        dest='pretrained_model',
        help='The directory for pretrained model',
        type=str,
        default='vgg16.pdparams')

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=10)

    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='The directory for train dataset',
        type=str,
        default='/Users/alex/baidu/HED-BSDS')

    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=1e-4)

    parser.add_argument(
        '--log_interval',
        dest='log_interval',
        help='Display logging information at every log_interval iter',
        default=10,
        type=int)

    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')

    parser.add_argument(
        '--resume-from',
        dest='resume_from',
        help='the checkpoint file to resume from'
    )

    return parser.parse_args()


def get_config(args):
    config = {}

    config['epochs'] = args.epochs

    config['pretrained_model'] = args.pretrained_model

    config['batch_size'] = args.batch_size
    config['dataset_root'] = args.dataset

    config['learning_rate'] = args.learning_rate

    config['log_interval'] = args.log_interval

    config['save_interval'] = args.save_interval
    config['save_dir'] = args.save_dir

    config['resume'] = False
    if args.resume_from is not None:
        config['resume'] = True
        config['checkpoint'] = args.resume_from

    return config


def main():
    args = parse_args()
    config = get_config(args)
    print(config)

    epochs = config['epochs']
    pretrained_model = config['pretrained_model']
    batch_size = config['batch_size']
    dataset_root = config['dataset_root']
    learning_rate = config['learning_rate']

    log_interval = config['log_interval']
    save_dir = config['save_dir']

    resume = config['resume']

    start_epoch = 0

    transforms = [
        Resize(target_size=(400, 400)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomDistort(
            brightness_range=0.4,
            contrast_range=0.4,
            saturation_range=0.4,
        ),
        # Normalize(mean=(104.00699, 116.66876762, 122.67891434), std=(57.375 ,57.12, 58.395))
        Normalize(mean=(104.00699, 116.66876762, 122.67891434), std=(1, 1, 1))
    ]
    dataset = Dataset(
        transforms=transforms,
        dataset_root=dataset_root,
        train_path=os.path.join(dataset_root, "train_pair.lst"))

    batch_sampler = paddle.io.DistributedBatchSampler(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        return_list=True,
    )

    bce_loss = BCELoss(weight="dynamic")
    model = HED(backbone_pretrained=pretrained_model)

    learning_rate = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=learning_rate, decay_steps=epochs * len(loader), power=0.9, end_lr=1e-8)
    lr = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=10000,
        start_lr=0,
        end_lr=1e-4)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr, parameters=model.parameters(), weight_decay=2e-4)

    callbacks = [
        LogPrinter(model, log_interval=log_interval, log_dir=save_dir),
        Checkpointer(model, optimizer, save_dir=save_dir)
    ]
    hooks = ComposeCallback(callbacks)

    if resume:
        checkpoint_path = config['checkpoint']
        checkpoint = paddle.load(checkpoint_path)
        model.set_state_dict(checkpoint['model_state_dict'])
        optimizer.set_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    train(model, loader, optimizer, loss_fn=bce_loss, hooks=hooks,
          start_epoch=start_epoch, epochs=epochs)


if __name__ == '__main__':
    main()
