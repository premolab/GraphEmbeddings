from datetime import datetime
import sys

def log_progress(sequence, every=None, size=None, add_val=False):
    start = datetime.now()
    try:
        from ipywidgets import IntProgress, HTML, VBox
        from IPython.display import display

        is_iterator = False
        if size is None:
            try:
                size = len(sequence)
            except TypeError:
                is_iterator = True
        if size is not None:
            if every is None:
                if size <= 200:
                    every = 1
                else:
                    every = size / 200  # every 0.5%
        else:
            assert every is not None, 'sequence is iterator, set every'

        if is_iterator:
            progress = IntProgress(min=0, max=1, value=1)
            progress.bar_style = 'info'
        else:
            progress = IntProgress(min=0, max=size, value=0)
        label = HTML()
        box = VBox(children=[label, progress])
        display(box)

        index = 0
        try:
            for index, record in enumerate(sequence, 1):
                if index == 1 or index % every == 0:
                    if is_iterator:
                        label.value = '{index} / ?'.format(index=index)
                    else:
                        N = len(sequence)
                        time = datetime.now()
                        progress.value = index
                        label.value = u'{index} / {size}'.format(
                            index=index,
                            size=size
                        )
                        label.value += "\rProgress: {:.2f}% | ETA/Total: {:.2f}/{:.2f} sec {}".format(100.0 * (index + 1) / N,
                                (time - start).seconds / (index + 1) * (N - 1 - index),
                                (time - start).seconds / (index + 1) * N, " " * 30)
                        last = time
                yield record
        except:
            progress.bar_style = 'danger'
            raise
        else:
            progress.bar_style = 'success'
            progress.value = index
            label.value = str(index or '?') + " Total: {:.2f} sec".format((datetime.now() - start).seconds)

    except:
            """
            display progress for loop list
            :param sequence: list
            :param every: minimal update iterval for progress
            :return: generator with progress output to stdout
            """
            N = len(sequence)
            template = "Progress: {:.2f}% | ETA/Total: {:.2f}/{:.2f} sec {}"
            last = start

            template2 = ('val: nan ' + template) if add_val else template

            sys.stdout.write(
                template2.format(0, float('nan'), float('nan'), ' '))
            for index, val in enumerate(sequence):
                yield val
                time = datetime.now()
                if (time - last).seconds > every:
                    template2 = ('val: {} '.format(val) + template) if add_val else template
                    sys.stdout.write('\r' + template2
                                     .format(100.0 * (index + 1) / N,
                                             (time - start).seconds / (index + 1) * (N - 1 - index),
                                             (time - start).seconds / (index + 1) * N, " " * 30))
                    sys.stdout.flush()
                    last = time

