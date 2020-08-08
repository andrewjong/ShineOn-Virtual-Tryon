import importlib


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace("_", "").lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print(
            f"In {module}, there should be a class whose name matches "
            f"{target_cls_name} in lowercase without underscore(_)"
        )
        exit(0)

    return cls
