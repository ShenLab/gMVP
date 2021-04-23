import time
import json
import argparse
import os
import sys
import logging
import shutil
from datetime import datetime
import glob
import random

from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr

import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import tensorflow as tf
import tensorflow_addons as tfa

#from optimization import create_optimizer

from model_attention import ModelAttention

from dataset import build_dataset
from loss import compute_loss
from constant_params import window_size, input_feature_dim

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.config.threading.set_intra_op_parallelism_threads(60)
tf.config.threading.set_inter_op_parallelism_threads(60)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging_formatter)
logger.addHandler(ch)


class LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, end_learning_rate, warmup_steps, decay_steps):
        super(LearningRate, self).__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        if decay_steps == 0:
            self.poly_decay_fn = lambda x: self.base_lr
        else:
            self.poly_decay_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                base_lr,
                decay_steps,
                end_learning_rate=end_learning_rate,
                power=1.0)

    def __call__(self, step):
        lr = tf.cond(
            step < self.warmup_steps, lambda: self.base_lr * tf.cast(
                step + 1, tf.float32) / tf.cast(self.warmup_steps, tf.float32),
            lambda: self.poly_decay_fn(step - self.warmup_steps))
        #if step % 100 == 0:
        #    tf.print('learning_rate', step, lr)

        return lr


class TestMetric(object):
    def __init__(self):
        self._targets = tf.zeros((0, ), tf.int32)
        self._preds = tf.zeros((0, ), tf.float32)

    def reset_states(self):
        self._targets = tf.zeros((0, ), tf.int32)
        self._preds = tf.zeros((0, ), tf.float32)

    def update_state(self, targets, preds):
        self._targets = tf.concat(
            [self._targets, tf.cast(targets, tf.int32)], axis=-1)
        self._preds = tf.concat(
            [self._preds, tf.cast(preds, tf.float32)], axis=-1)

    def result_auROC(self):
        try:
            auROC = roc_auc_score(self._targets.numpy(), self._preds.numpy())
            return auROC
        except:
            return 0.0

    def result_auPR(self):
        try:
            precision, recall, _ = precision_recall_curve(
                self._targets.numpy(), self._preds.numpy())
            auPR = auc(recall, precision)
            return auPR
        except:
            return 0.0

    def result_pvalue(self):
        all_pred = self._preds.numpy()
        all_label = self._targets.numpy()
        mtest = mannwhitneyu(all_pred[all_label == 1],
                             all_pred[all_label == 0],
                             alternative='two-sided')
        pvalue = mtest.pvalue
        return pvalue

    def result_total(self):
        res = self._targets.numpy()
        return res.shape[0]

    def result_neg(self):
        res = self._targets.numpy()
        return res.shape[0] - np.sum(res)

    def result_pos(self):
        res = self._targets.numpy()
        return np.sum(res)

    def result_corr(self):
        try:
            all_pred = self._preds.numpy()
            all_label = self._targets.numpy()
            corr, pvalue = spearmanr(all_pred, all_label)
            return corr, pvalue
        except:
            return 0.0

    def result_max(self):
        try:
            all_pred = self._preds.numpy()
            return np.max(all_pred)
        except:
            return 0.0


def train_single_gpu(config, args):
    #setup logger
    str_t = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_dir = f'./res/{str_t}'
    config['train']['train_dir'] = train_dir
    os.makedirs(train_dir)
    os.makedirs(train_dir + '/result')
    os.makedirs(train_dir + '/model')

    fh = logging.FileHandler(f'{train_dir}/train.log')
    fh.setFormatter(logging_formatter)
    logger.addHandler(fh)

    logger.info(json.dumps(config, indent=4))

    #train and validate files
    batch_size = config['train']['batch_size']

    input_config = config['input']
    train_files = glob.glob(input_config['train'])
    validate_files = glob.glob(input_config['validate'])
    test_files = glob.glob(input_config['test'])

    train_dataset = build_dataset(train_files, batch_size)
    validate_dataset = build_dataset(validate_files, batch_size)
    test_dataset = build_dataset(test_files, batch_size)

    #metric
    metric_train_loss = tf.keras.metrics.Mean(name='train_loss')

    #model
    model = ModelAttention(config['model'])
    if args.init_model is not None:
        x = tf.ones([batch_size, window_size, input_feature_dim],
                    dtype=tf.float32)
        ref_aa = tf.ones((batch_size, ), dtype=tf.int32)
        alt_aa = tf.ones((batch_size, ), dtype=tf.int32)
        mask = tf.zeros((batch_size, window_size), dtype=tf.float32)
        model((ref_aa, alt_aa, x), training=False, mask=mask)
        model.load_weights(args.init_model)

        #set other layers untrainable
        for layer in model.layers:
            if layer.name == 'gru_cell':
                layer.trainable = True
            else:
                layer.trainable = True
            print(layer.name, layer.trainable)
        #double check
        print('trainable variables')
        for var in model.trainable_variables:
            print(var.name)

    #learning rate
    init_learning_rate = config['train']['learning_rate']
    end_learning_rate = config['train']['end_learning_rate']

    warmup_steps, decay_steps = config['train']['warmup_steps'], config[
        'train']['decay_steps']

    learning_rate = LearningRate(init_learning_rate,
                                 end_learning_rate=end_learning_rate,
                                 warmup_steps=warmup_steps,
                                 decay_steps=decay_steps)

    #training algorithm
    opt = config['train'].get('opt', 'adam')
    if opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise NotImplementedError(f"opt {opt} not NotImplementedError")

    def _save_res(var_id, target, pred, name, epoch):
        with open(f'{train_dir}/result/epoch_{epoch}_{name}.score', 'w') as f:
            f.write('var\ttarget\tScore\n')
            for a, c, d in zip(var_id, target, pred):
                f.write('{}\t{:d}\t{:f}\n'.format(a.numpy().decode('utf-8'),
                                                  int(c), d))
        return True

    @tf.function(input_signature=[train_dataset.element_spec])
    def test_step(sample):
        var, ref_aa, alt_aa, feature, label, padding_mask = sample

        logit = model((ref_aa, alt_aa, feature), False, padding_mask)

        loss = compute_loss(label, logit)

        pred = model.predict_from_logit(logit)

        return var, label, pred, loss

    def test(test_dataset,
             data_name,
             epoch,
             auc=False,
             pvalue=False,
             corr=False):
        #metrics
        metric_test = TestMetric()
        metric_test_loss = tf.keras.metrics.Mean(name='test_loss')

        metric_test.reset_states()
        metric_test_loss.reset_states()

        all_pred, all_label, all_var = [], [], []

        for step, sample in enumerate(test_dataset):
            var, label, pred, loss = test_step(sample)
            metric_test.update_state(label, pred)
            metric_test_loss.update_state(loss)

            all_pred.extend(list(pred))
            all_label.extend(list(label))
            all_var.extend(list(var))

        all_var = np.array(all_var)
        all_label = np.array(all_label)
        all_pred = np.array(all_pred)

        _save_res(all_var, all_label, all_pred, data_name, epoch)

        if auc:
            logger.info(
                f'{data_name}  pos= {metric_test.result_pos()} neg= {metric_test.result_neg()} loss= {metric_test_loss.result()} auPR= {metric_test.result_auPR()} auROC= {metric_test.result_auROC()} max= {metric_test.result_max()}'
            )
        if pvalue:
            logger.info(
                f'{data_name}  pos= {metric_test.result_pos()} neg= {metric_test.result_neg()} loss= {metric_test_loss.result()} pvalue= {metric_test.result_pvalue()}'
            )

        if corr:
            corr, pvalue = metric_test.result_corr()
            logger.info(
                f'{data_name}  pos= {metric_test.result_total()} corr= {corr} pvalue= {pvalue} max= {metric_test.result_max()}'
            )

        return metric_test.result_auROC()

    @tf.function(input_signature=[train_dataset.element_spec])
    def train_step(sample):
        var, ref_aa, alt_aa, feature, label, padding_mask = sample
        with tf.GradientTape() as tape:
            logit = model((ref_aa, alt_aa, feature), True, padding_mask)
            loss = compute_loss(label, logit)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metric_train_loss.update_state(loss)
        #if optimizer.iterations % 512 == 0:
        #    _update_gradient_norm_summary(model.trainable_variables, gradients)

        return loss

    print('init model')
    #test(test_dataset, 'test', 0, pvalue=False, auc=True, corr=False)

    EPOCHS = 512
    watch_auROC = -1.0
    watch_epoch = -1
    patience_epochs = 5
    for epoch in range(EPOCHS):
        start = time.time()

        for step, samples in enumerate(train_dataset):
            loss = train_step(samples)

            #model summary
            if optimizer.iterations == 1:
                model.summary(print_fn=logger.info)

        logger.info(f'Epoch {epoch} Loss {metric_train_loss.result():.4f}')
        metric_train_loss.reset_states()

        model.save_weights(f'{train_dir}/model/epoch-{epoch}.h5')

        #validate and test
        validate_auROC = test(validate_dataset,
                              'validate',
                              epoch,
                              pvalue=False,
                              auc=True,
                              corr=False)
        if validate_auROC > watch_auROC:
            watch_auROC = validate_auROC
            watch_epoch = epoch

        if epoch - watch_epoch == patience_epochs:
            logger.info(
                f'best_epoch {watch_epoch} best_validate_auROC= {watch_auROC}')
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--init_model', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_single_gpu(config, args)


if __name__ == '__main__':
    main()
