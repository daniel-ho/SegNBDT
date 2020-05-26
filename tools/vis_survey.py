import os
from jinja2 import Template
import sys
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--baseline', nargs='*')
parser.add_argument('--baseline-original', nargs='*')
parser.add_argument('--ours', nargs='*')
args = parser.parse_args()

# attempts to match stem
stem_to_path = {}
for a in args.baseline:
    stem_to_path[Path(a).stem] = [a]

for a2 in args.baseline_original:
    stem_to_path[Path(a2).stem].append(a2)

for b in args.ours:
    stem = Path(b).stem
    cls = stem.split('-')[6]
    stem = stem.replace(f'-{cls}', '', 1)
    if stem in stem_to_path:
        stem_to_path[stem].extend((cls, b))

files = [value for value in stem_to_path.values() if len(value) == 4]

template = Template(
"""
<html>
  <head>
    <title>Figure Browser</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
    <script
      src="https://code.jquery.com/jquery-3.5.1.min.js"
      integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0="
      crossorigin="anonymous"></script>
    <script>
var index = -1;
var files = [
    {% for file in files %}
        {baseline: '{{ file[0] }}', baseline_original: '{{ file[1] }}', cls: '{{ file[2] }}', ours: '{{ file[3] }}'},
    {% endfor %}
];

function select(i, dontPushHistory) {
  i = parseInt(i);
  index = i;
  $('#curr').html(i);

  var el_a = $('div#figure-a');
  var el_b = $('div#figure-b');

  if (i % 2 == 0) {
    tmp = el_a;
    el_a = el_b;
    el_b = tmp;
  }

  var file = files[i - 1];
  el_a.html('<div class="row"><div class="col"><img src="' + file['baseline_original'] + '"><img class="cover original-cover" src="original.png"><p><b>1. Start here</b><br/>Goal: Classify center</br> pixel.</p></div><div class="col"><img src="' + file['baseline'] + '"><img class="cover final-cover" src="final.png"><p><b>2. Prediction<b><br/>' + file['cls'] + '</p></div></div>');
  el_b.html('Loading...');
  el_b.load(file['ours'], function() {
    console.log('Loaded ' + i);

    if (window.onload) {
        window.onload();
    }
  })

  if (i <= 1) {
    $('#prev').hide();
  } else {
    var url = '?id=' + (i-1);
    $('#prev').show().attr('i', i-1).attr('href', '?id=' + (i-1));
    $('#prev span').html(i - 1);
  }

  if (i >= files.length) {
    $('#next').hide();
  } else {
    var url = '?id=' + (i+1);
    $('#next').show().attr('i', i+1).attr('href', url);
    $('#next span').html(i + 1);
  }

  if (!dontPushHistory) {
    history.pushState('data', '', '?id=' + i);
  }
}

function onLoadSelect() {
  const urlParams = new URLSearchParams(window.location.search);
  const i = urlParams.get('id') || 1;
  select(i, true);
}

$(document).ready(function() {
  $('#total').html(files.length);
  onLoadSelect();

  $('#next').on('click', function(e) {
      select($('#next').attr('i'));
      e.preventDefault();
  });

  $('#prev').on('click', function(e) {
      select($('#prev').attr('i'));
      e.preventDefault();
  });

  $(window).on('popstate', onLoadSelect);
});
    </script>
    <style>
header {
    font-size:1.25em;
    padding:0.1em 0;
    position:fixed;
    top:0;
    left:0;
    width:100%;
    box-shadow: 0 0 1em rgba(0,0,0,0.3);
    z-index:100;
    background-color:#FFF;
    display:flex;
    justify-content:space-between;
}
header img {
    margin-top:14px;
    width:300px;
    height:30px;
    border:2px solid #cccccc;
    margin-right:20px;
}
p, h1 {
    font-family:"Cormorant Garamond";
}
.comparison {
    display:flex;
    flex-direction:row;
    margin-top: 5.5em;
}
.explanation {
    flex:1;
    padding:0.25em 2.5em;
}
.explanation + .explanation {
border-left:2px solid #eee;
}
h1 {
    margin:0;
    padding:0;
}
.row {
    margin-top:0.25em;
    display:flex;
    flex-direction:row;
}
.col {
    position:relative;
}
.col .cover {
    position:absolute;
    top:0;
    left:0;
}
.col img {
    max-width:150px;
}
.col + .col {
    margin-left:0.25em;
}
.curr {
margin:0 1em;
}
#prev, #next {
    background-color:#eee;
    margin-top:-0.5em;
    padding:0.5em;
    border-radius:0.2em;
    color:#666;
    text-decoration:none;
}
#prev:hover, #next:hover {
    color:#000;
    background-color:#ccc;
}
.nav {
margin-left:20px;
}
.colorbar {
display:flex;
}
.colorbar span {
margin-top:25px;
font-size:0.8em;
margin-right:10px;
}
    </style>
  </head>
  <body>
    <header>
      <p class="nav">
        <a href="#" id="prev">&laquo; prev (<span>-1</span>)</a>
        <span class="curr"><b id="curr">-1</b> of <span id="total">-1</span></span>
        <a href="#" id="next">next (<span>-1</span>) &raquo;</a>
      </p>
      <div class="colorbar">
      <span>Pixel Importance (applies to both explanations):</span>
      <img src="colormap.png">
      </div>
    </header>
    <div class="comparison">
        <div class="explanation">
        <h1>Explanation A</h1>
            <div id="figure-a"></div>
        </div>
        <div class="explanation">
            <h1>Explanation B</h1>
            <div id="figure-b"></div>
        </div>
    </div>
  </body>
</html>

""")

with open("index.html", "w") as f:
    f.write(template.render(files=files))
