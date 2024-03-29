import os

from .abstract_image_provider import AbstractImageProvider


class ReadingImageProvider(AbstractImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, image_suffix=None, has_alpha=False):
        super(ReadingImageProvider, self).__init__(image_type, fn_mapping, has_alpha=has_alpha)
        
        # Ensure paths is a directory
        if not os.path.isdir(paths):
            raise ValueError(f"The provided 'paths' is not a valid directory: {paths}")

        self.paths = paths

        # List all files in the directory
        all_files = os.listdir(paths)

        if image_suffix is not None:
            self.im_names = [n for n in all_files if image_suffix in n.lower() and n.lower().endswith('.png')]
        else:
            self.im_names = [n for n in all_files if n.lower().endswith('.png')]

    def get_indexes_by_names(self, names):
        return [idx for idx, name in enumerate(self.im_names) if os.path.splitext(name)[0] in names]

    def __getitem__(self, item):
        return self.image_type(self.paths, self.im_names[item], self.fn_mapping, self.has_alpha)

    def __len__(self):
        return len(self.im_names)