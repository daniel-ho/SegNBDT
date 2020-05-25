import os
from jinja2 import Template
import glob
import argparse
from collections import defaultdict
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='*', help='Explicitly list dirs')
    parser.add_argument('--row', choices=('sort', 'match'),
                        default='sort')
    args = parser.parse_args()


def paths_from_directories(directories, mode='match'):
    paths_per_dirs = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        paths_per_dirs.append(sorted(glob.iglob(f'{directory}/*.jpg'), key=lambda s: int(Path(s).stem.split('-')[1])))

    if mode == 'sort':
        paths_per_rows = list(zip(*paths_per_dirs))
    elif mode == 'match':
        name_to_paths = defaultdict(lambda: [])
        for paths_per_dir in paths_per_dirs:
            for path in paths_per_dir:
                path = Path(path)
                name_to_paths[path.name].append(str(path))

        n = max([len(paths) for paths in name_to_paths.values()])
        paths_per_rows = [paths for paths in name_to_paths.values() if len(paths) == n]
    else:
        raise UserWarning(f'{args.include} is not a valid include mode.')
    return paths_per_rows


if __name__ == '__main__':
    args = get_args()
    directories = list(sorted(args.directory or os.listdir('.')))
    print(directories)

    paths_per_rows = paths_from_directories(directories, args.row)

    template = Template('''
    <html>
        <body>
        <table border="0" style="table-layout: fixed;">
            <tr>
                {% for header in headers %}
                <th>{{ header }}</th>
                {% endfor %}
            </tr>
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
        f.write(template.render(
            headers=directories,
            paths_per_rows=paths_per_rows,
            width=300))
