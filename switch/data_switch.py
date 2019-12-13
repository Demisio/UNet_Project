
def data_switch(data_identifier):

    # Get Data
    if data_identifier == 'heart':
        from data_processing.heart_data import heart_data as data_loader
    else:
        raise ValueError('Unknown data identifier: %s' % data_identifier)

    return data_loader