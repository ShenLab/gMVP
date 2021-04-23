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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    input_base_dir = input_config['base_dir']
    all_files = glob.glob(input_base_dir + '/' + input_config['train'][:-1] +
                          args.random + '*tfrec')
    #all_files = glob.glob('../dataset/tf/f_v1_w64_2021_v2' + '/' +
    #                      input_config['train'][:-1] + args.random + '*tfrec')
    random.seed(2020)
    random.shuffle(all_files)
    train_files, validate_files = [], []
    for i in range(10):
        if i == args.cv:
            validate_files.append(all_files[i])
        else:
            train_files.append(all_files[i])

    print(train_files)
    print(validate_files)

    asd = glob.glob(input_base_dir + '/' + 'ASD' + '.tfrec')
    ndd = glob.glob(input_base_dir + '/' + 'NDD' + '.tfrec')
    control = glob.glob(input_base_dir + '/' + 'Control' + '.tfrec')
    brca2 = glob.glob(input_base_dir + '/' + 'BRCA2' + '.tfrec')
    pparg = glob.glob(input_base_dir + '/' + 'PPARG' + '.tfrec')
    #train_files += pparg

    train_dataset = build_dataset(train_files, batch_size)
    validate_dataset = build_dataset(validate_files, batch_size)

    #model
    model_type = config['train']['model_type']
    if model_type == 'attention':
        model = ModelAttention(config['model'])
    else:
        raise ValueError(f'model type {model_type} does not exist.')
    #learning rate
    init_learning_rate = config['train']['learning_rate']
    end_learning_rate = config['train']['end_learning_rate']
    '''
    warmup_epochs = config['train']['warmup_epochs']
    decay_epochs = config['train']['decay_epochs']

    training_samples = 0
    for inputs in train_dataset:
        training_samples += inputs[0].shape[0]
    logger.info(f'training_samples= {training_samples}')

    batches_each_epoch = int(training_samples / batch_size)
    warmup_steps = batches_each_epoch * warmup_epochs
    decay_steps = batches_each_epoch * decay_epochs
    '''

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
        #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    elif opt == 'adamw':
        weight_decay_rate = config['train']['weight_decay_rate']
        optimizer = tfa.optimizers.AdamW(
            weight_decay=weight_decay_rate,
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )
        '''
        optimizer = create_optimizer(init_learning_rate,
                                     decay_steps + warmup_steps,
                                     warmup_steps,
                                     end_lr=end_learning_rate,
                                     optimizer_type='adamw')
        '''

    else:
        raise NotImplementedError(f"opt {opt} not NotImplementedError")

    #metrics
    metric_train_loss = tf.keras.metrics.Mean(name='train_loss')
    metric_test_loss = tf.keras.metrics.Mean(name='test_loss')
    metric_test = TestMetric()

    #summary
    train_log_dir = f'{train_dir}/summary/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def _update_histogram_summary():
        with train_summary_writer.as_default():
            for var in model.trainable_variables:
                if 'kernel:' in var.name or 'gamma:' in var.name or 'beta:' in var.name:
                    tf.summary.histogram(var.name,
                                         var,
                                         step=optimizer.iterations)

    def _update_gradient_norm_summary(var, grad):
        with train_summary_writer.as_default():
            for v, g in zip(var, grad):
                if 'kernel:' in v.name or 'gamma:' in v.name or 'beta:' in v.name:
                    tf.summary.scalar(f'gradient_norm/{v.name}',
                                      tf.norm(g, ord='euclidean'),
                                      step=optimizer.iterations)

    @tf.function(input_signature=[validate_dataset.element_spec])
    def test_step(sample):
        var, ref_aa, alt_aa, feature, label, padding_mask = sample

        logit = model((ref_aa, alt_aa, feature), False, padding_mask)

        loss = compute_loss(label, logit)

        pred = model.predict_from_logit(logit)

        return var, label, pred, loss

    def _save_res(var_id, target, pred, name, epoch):
        with open(f'{train_dir}/result/epoch_{epoch}_{name}.score', 'w') as f:
            f.write('var\ttarget\tScore\n')
            for a, c, d in zip(var_id, target, pred):
                f.write('{}\t{:d}\t{:f}\n'.format(a.numpy().decode('utf-8'),
                                                  int(c), d))
        return True

    def test(test_dataset,
             data_name,
             epoch,
             auc=False,
             pvalue=False,
             corr=False):
        metric_test_loss.reset_states()
        metric_test.reset_states()

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

        return metric_test_loss.result()

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

    EPOCHS = 512
    watch_loss = 10000.0
    watch_epoch = -1
    patience_epochs = 5
    for epoch in range(EPOCHS):
        start = time.time()

        for step, samples in enumerate(train_dataset):
            loss = train_step(samples)
            #tf.print(
            #    f'lr= {learning_rate(global_step)} wd={weight_decay(global_step)}'
            #)

            #model summary
            if optimizer.iterations == 1:
                model.summary(print_fn=logger.info)

            #logging kernel weights
            #if (optimizer.iterations + 1) % 512 == 0:
            #    _update_histogram_summary()

        logger.info(f'Epoch {epoch} Loss {metric_train_loss.result():.4f}')
        metric_train_loss.reset_states()

        model.save_weights(f'{train_dir}/model/epoch-{epoch}.h5')

        #validate and test
        validate_loss = test(validate_dataset,
                             'validate',
                             epoch,
                             pvalue=False,
                             auc=True,
                             corr=False)
        if validate_loss < watch_loss:
            watch_loss = validate_loss
            watch_epoch = epoch

        #denovo
        if epoch - watch_epoch == patience_epochs:
            logger.info(f'best_epoch {watch_epoch} min_loss= {watch_loss}')
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--random', type=str, default='0')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_single_gpu(config, args)


if __name__ == '__main__':
    main()
