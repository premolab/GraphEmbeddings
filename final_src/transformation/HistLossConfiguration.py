from parse import parse

from transformation.RunConfiguration import RunConfiguration


class HistLossConfiguration:
    def __init__(self,
                 metric,
                 simmatrix_method,
                 loss_method,
                 calc_pos_method,
                 calc_neg_method,
                 calc_hist_method,
                 batch_size):
        self.metric = metric
        self.simmatrix_method = simmatrix_method
        self.loss_method = loss_method
        self.calc_pos_method = calc_pos_method
        self.calc_neg_method = calc_neg_method
        self.calc_hist_method = calc_hist_method
        self.batch_size = batch_size

    def __str__(self):
        return '{}_{}_{}_{}_{}_{}_{}'.format(self.metric,
                                             self.simmatrix_method,
                                             self.loss_method,
                                             self.calc_pos_method,
                                             self.calc_neg_method,
                                             self.calc_hist_method,
                                             self.batch_size)

    @staticmethod
    def from_string(s):
        return HistLossConfiguration(*parse('{}_{}_{}_{}_{}_{}_{}', s))

    @staticmethod
    def from_run_configuration(run_configuration: RunConfiguration):
        assert run_configuration.method.startswith('hist_loss_')
        return HistLossConfiguration.from_string(run_configuration.method[10:])
