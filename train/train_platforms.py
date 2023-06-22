import os
import re

try:
    # The import command must be at the top of the file and NOT internal to the ClearmlPlatform class,
    # because when cloning, the 'patching' procedure is done by the clearml engine at the very beginning
    from clearml import Task
except:
    pass


class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_media(self, title, series, iteration, local_path):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        path, name = os.path.split(save_dir)
        if len(re.findall('\d\d\d\d', name)) == 1:  # special care for benchmarks
            name = os.path.split(path)[1] + "-" + name
        self.task = Task.init(project_name='sin_mdm',
                              task_name=name)
        print('self.task.get_parameters_as_dict() : ')
        print(self.task.get_parameters_as_dict())
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_media(self, title, series, iteration, local_path):
        self.logger.report_media(title=title, series=series, iteration=iteration, local_path=local_path)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass

