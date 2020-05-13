"""Generates HTML visualization.

In this visualization, your cursor position is discretized to a 50x50 bin, and
the corresponding image is shown.

To use, output saliency maps to a directory of your choosing
(by default, ./images, from whatever directory you're in). This script will
iterate over all images to produce the vis. Note this script assumes images
come in the following format:

    normgrad-image-1-pixel_i-0-pixel_j-0-layer-stage2.0.fuse_layers.1.0.0.1.png

You can download our saliency outputs from:

    TBD
"""

import glob
from jinja2 import Template
import random

paths = []
for path in glob.iglob('./images/*'):
    parts = path.split('-')
    paths.append({
        'src': path,
        'i': int(parts[4]),
        'j': int(parts[6])
    })

template = '''
<html>
  <head>
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
<style>
body, html {
  padding:0;
  margin:0;
}
body, html, .img-container {
  display:flex;
}
.img-container img {
  width:100%;
}
body {
  flex-direction:column;
}
</style>
  </head>
  <body>

  {% for path in paths %}
  <div class="img-container">
    <img src="{{ path['src'] }}" i={{ path['i'] }} j={{ path['j'] }}>
  </div>
  {% endfor %}
    <script>
$('img').hide();

var imgWidth = 2048;
var width = $('html').width();
var ratio = width / imgWidth;

$('html').mousemove(function (e) {
    var j = Math.round(e.pageX / 50 / ratio) * 50;
    var i = Math.round(e.pageY / 50 / ratio) * 50;

    var target = $('img[i=' + i + '][j=' + j + ']');
    if (target.length > 0) {
      $('img').hide();
      $('img[i=' + i + '][j=' + j + ']').show();
    }

    console.log(i, j);
});
    </script>
  </body>
</html>
'''

html = template.render(paths=paths, random=random.random())
with open('index.html', 'w') as f:
    f.write(html)
