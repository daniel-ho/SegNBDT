"""Generates HTML visualization.

In this visualization, your cursor position is discretized to a 50x50 bin, and
the corresponding image is shown.

To use, output saliency maps to a directory of your choosing
(by default, ./images, from whatever directory you're in). This script will
iterate over all images to produce the vis. Note this script assumes images
come in the following format:

    normgrad-image-1-pixel_i-0-pixel_j-0-layer-stage2.0.fuse_layers.1.0.0.1.png

You can download our saliency outputs from:

    https://drive.google.com/file/d/16rNpHXr8ZVshh0YAWxqxGC_A-JDS9qeJ/view?usp=sharing

Here is the script we used to produce saliency maps:

    # We ran into memory issues every 200 passes. Rather than fix it, we just ran
    # multiple runs of 200 passes each :P
    for method in NormGrad GradCAM;
    do
        for i in 0 200 400 600 800 1000 1200 1400 1600 1800 2000;
        do CUDA_VISIBLE_DEVICES=4 python tools/vis_gradcam.py \
            --cfg experiments/cityscapes/vis/vis_seg_hrnet_w18_small_v1_512x1024.yaml \
            --vis-mode ${method} \
            --image-index 1 \
            --pixel-i-range 0 1023 25 \
            --pixel-j-range ${i} $(expr $i + 200) 25 \
            --pixel-cartesian-product \
            --target-layers stage4.0.fuse_layers.2.1.0.1,stage3.0.fuse_layers.2.1.0.1,stage2.0.fuse_layers.1.0.0.1 \
            TEST.MODEL_FILE pretrained_models/hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.pth
        done;
    done;
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

template = Template('''
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
''')

html = template.render(paths=paths, random=random.random())
with open('index.html', 'w') as f:
    f.write(html)
