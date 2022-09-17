import json
import logging
import os
import time
from collections import OrderedDict

import paddle

from utils import get_root_logger


class Callback(object):
    def __init__(self, model):
        self.model = model

    def on_step_begin(self, status):
        pass

    def on_step_end(self, status):
        pass

    def on_epoch_begin(self, status):
        pass

    def on_epoch_end(self, status):
        pass


class ComposeCallback(object):
    def __init__(self, callbacks):
        callbacks = [callback for callback in list(callbacks) if callback is not None]
        for callback in callbacks:
            assert isinstance(callback, Callback), \
                f"callback should be a sub class of Callback, but got {type(callback)}"
        self.callbacks = callbacks

    def on_step_begin(self, status):
        for callback in self.callbacks:
            callback.on_step_begin(status)

    def on_step_end(self, status):
        for callback in self.callbacks:
            callback.on_step_end(status)

    def on_epoch_begin(self, status):
        for callback in self.callbacks:
            callback.on_epoch_begin(status)

    def on_epoch_end(self, status):
        for callback in self.callbacks:
            callback.on_epoch_end(status)


class LogPrinter(Callback):
    def __init__(self, model, log_interval=100, log_dir=None):
        super(LogPrinter, self).__init__(model)

        self.log_interval = log_interval

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(log_dir, f'{timestamp}.log')
        self.logger = get_root_logger(log_file=log_file, log_level=logging.INFO)

        self.json_log_path = os.path.join(log_dir, f"{timestamp}.log.json")

    def on_step_end(self, status):
        epoch_id = status['epoch_id']
        step_id = status['step_id']
        steps_per_epoch = status['steps_per_epoch']
        loss = status['loss']
        lr = status['learning_rate']

        if step_id % self.log_interval == 0:
            self.logger.info(
                "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}".format(epoch_id, step_id, steps_per_epoch,
                                                                                  loss, lr))
            log_dict = OrderedDict(epoch=epoch_id, iter=step_id, lr=lr, loss=loss)
            self._dump_json_log(log_dict, self.json_log_path)

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def _dump_json_log(self, log_dict, json_log_path):
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)

        with open(json_log_path, 'a+') as f:
            json.dump(json_log, f)
            f.write('\n')


class Checkpointer(Callback):
    def __init__(self, model, optimizer, save_dir):
        super(Checkpointer, self).__init__(model)

        self.optimizer = optimizer
        self.save_dir = save_dir

    def on_epoch_end(self, status):
        epoch_id = status['epoch_id']
        loss = status['loss']

        path = os.path.join(self.save_dir, "epoch_{}.pth".format(epoch_id + 1))
        paddle.save({
            'epoch': epoch_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, path)
