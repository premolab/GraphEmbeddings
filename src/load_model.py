

def get_method_name(metric='EMD',
                    simmatrix_method='ID',
                    loss_method='ASIM',
                    calc_pos_method='NORMAL',
                    calc_neg_method='NORMAL',
                    calc_hist_method='NORMAL',
                    batch_size=0):
    method_name = 'hist_loss'
    if metric != 'NORMAL':
        method_name += '_' + metric

    method_name += '_' + simmatrix_method
    method_name += '_' + loss_method

    if calc_pos_method != 'NORMAL':
        method_name += '_' + calc_pos_method

    if calc_neg_method != 'NORMAL':
        method_name += '_' + calc_neg_method

    if calc_hist_method != 'NORMAL':
        method_name += '_' + calc_hist_method

    if batch_size != 0:
        method_name += '_BATCH_' + str(batch_size)

    return method_name
