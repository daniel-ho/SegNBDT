import os
from jinja2 import Template
import glob

directories = sorted(os.listdir('.'))

paths_per_dirs = []
for directory in directories:
    if not os.path.isdir(directory):
        continue
    paths_per_dirs.append(sorted(glob.iglob(f'{directory}/*.jpg'), key=lambda s: int(s.split('-')[1])))

paths_per_ims = list(zip(*paths_per_dirs))

template = Template('''
<html>
    <body>
    <table border="0" style="table-layout: fixed;">
    {% for paths_per_im in paths_per_ims %}
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
    f.write(template.render(paths_per_ims=paths_per_ims, width=300))
