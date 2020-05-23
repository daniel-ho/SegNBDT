import os
from jinja2 import Template
import glob
import argparse
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--row', choices=('sort', 'match'),
                    default='sort')
args = parser.parse_args()


directories = sorted(os.listdir('.'))

paths_per_dirs = []
for directory in directories:
    if not os.path.isdir(directory):
        continue
    paths_per_dirs.append(sorted(glob.iglob(f'{directory}/*.jpg'), key=lambda s: int(s.split('-')[1])))

if args.row == 'sort':
    paths_per_rows = list(zip(*paths_per_dirs))
elif args.row == 'match':
    name_to_paths = defaultdict(lambda: [])
    for paths_per_dir in paths_per_dirs:
        for path in paths_per_dir:
            path = Path(path)
            name_to_paths[path.name].append(str(path))

    n = max([len(paths) for paths in name_to_paths.values()])
    paths_per_rows = [paths for paths in name_to_paths.values() if len(paths) == n]
else:
    raise UserWarning(f'{args.include} is not a valid include mode.')


template = Template('''
<html>
    <body>
    <table border="0" style="table-layout: fixed;">
    {% for paths_per_im in paths_per_rows %}
        <tr>
        {% for path in paths_per_im %}
            <td halign="center" style="word-wrap: break-word;" valign="top">
              <a href="{{ path }}">
                <img src="{{ path }}" width="{{ width }}px"></img>
              </a>
            </td>
        {% endfor %}
        </tr>
    {% endfor %}
    </body>
</html>
''')

with open('index.html', 'w') as f:
    f.write(template.render(paths_per_rows=paths_per_rows, width=300))
