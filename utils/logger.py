
import logging
import tensorboard_logger as tb_logger
from generic_utils import Progbar

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
tb_logger.configure(opt.logger_name, flush_secs=5)

# logger = logging.getLogger(__file__)
# logging.basicConfig(
#     format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
#     datefmt='%d %b %H:%M:%S')
# logger.setLevel(logging.INFO)

tb_logger.log_value('epoch', epoch, step=model.Eiters)
tb_logger.log_value('step', i, step=model.Eiters)
tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
tb_logger.log_value('data_time', data_time.val, step=model.Eiters)

# tensorboard --logdir=path/to/log-directory

def log(logging=print):
    logging('Test: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            .format(i, len(data_loader), batch_time=batch_time))


progbar = Progbar(train_loader.dataset.length)
progbar.add(b_size, values=[("loss", loss)])



