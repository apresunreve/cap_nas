"""
Minimal restored ModelLoader implementation for PlainNet/MasterNet training.

Most source files for the original ModelLoader package are absent in this
worktree, but the active NASevo training path only needs the
``/path/Masternet.py:MasterNet`` architecture loader.
"""

import importlib.util
import os


def _get_model_(arch, num_classes, pretrained=False, opt=None, argv=None):
    if arch.find('.py:MasterNet') >= 0:
        module_path, class_name = arch.split(':', 1)
        assert class_name == 'MasterNet'

        if not os.path.isabs(module_path):
            working_dir = os.path.dirname(os.path.dirname(__file__))
            module_path = os.path.join(working_dir, module_path)
        if not os.path.exists(module_path) and os.path.exists(module_path + 'c'):
            module_path = module_path + 'c'

        spec = importlib.util.spec_from_file_location('AnyPlainNet', module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Unable to import MasterNet from {module_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print('Create model: {}'.format(arch))
        return module.MasterNet(num_classes=num_classes, opt=opt, argv=argv)

    raise ValueError('Unknown model arch: ' + arch)


def get_model(opt, argv):
    return _get_model_(arch=opt.arch, num_classes=opt.num_classes, opt=opt, argv=argv)
